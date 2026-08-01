# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded scheduling for compatible streaming Code2Wav chunks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Hashable


@dataclass(frozen=True)
class Code2WavAdmission:
    request: Any
    target_status: Any


@dataclass(frozen=True)
class _PendingChunk:
    request: Any
    target_status: Any
    key: Hashable
    ready_at: float


class Code2WavMicrobatchScheduler:
    """Collect compatible ready chunks without blocking the model worker.

    The scheduler owns no request queues and performs no I/O.  ``offer``
    admits a request immediately when batching is disabled, or returns a
    complete group when a matching request arrives.  ``poll`` releases groups
    whose bounded wait has expired.  The transfer adapter owns queue routing
    and request lifecycle cleanup.
    """

    def __init__(self, *, max_batch_size: int = 0, wait_ms: float = 0.0):
        self.max_batch_size = max(0, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_ms)) / 1000.0
        self._pending: dict[Hashable, deque[_PendingChunk]] = defaultdict(deque)
        self._request_keys: dict[str, Hashable] = {}
        self._stats = {
            "offers": 0,
            "pending": 0,
            "matched_b2": 0,
            "matched_requests": 0,
            "deadline_b1": 0,
            "deadline_requests": 0,
            "key_mismatch": 0,
            "cancelled": 0,
            "max_ready_skew_ms": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return self.max_batch_size > 1 and self.wait_seconds > 0.0

    def offer(
        self,
        request: Any,
        target_status: Any,
        key: Hashable,
        now: float,
    ) -> list[Code2WavAdmission]:
        self._stats["offers"] += 1
        if not self.enabled:
            return [Code2WavAdmission(request, target_status)]

        request_id = str(request.request_id)
        self.cancel(request_id)
        if self._pending and key not in self._pending:
            self._stats["key_mismatch"] += 1
        pending = self._pending[key]
        pending.append(_PendingChunk(request, target_status, key, now))
        self._request_keys[request_id] = key
        self._stats["pending"] = len(self._request_keys)
        if len(pending) < self.max_batch_size:
            return []
        return self._pop_group(key, matched=True)

    def poll(self, now: float) -> list[list[Code2WavAdmission]]:
        if not self.enabled:
            return []
        groups: list[list[Code2WavAdmission]] = []
        for key, pending in list(self._pending.items()):
            if pending and now - pending[0].ready_at >= self.wait_seconds:
                groups.append(self._pop_group(key, matched=False))
        return groups

    def cancel(self, request_id: str) -> None:
        key = self._request_keys.pop(str(request_id), None)
        if key is None:
            return
        self._stats["cancelled"] += 1
        pending = self._pending.get(key)
        if pending is None:
            return
        self._pending[key] = deque(item for item in pending if str(item.request.request_id) != str(request_id))
        if not self._pending[key]:
            self._pending.pop(key, None)

    def contains(self, request_id: str) -> bool:
        return str(request_id) in self._request_keys

    def pending_count(self) -> int:
        return len(self._request_keys)

    def stats_snapshot(self) -> dict[str, int | float]:
        snapshot = dict(self._stats)
        snapshot["pending"] = len(self._request_keys)
        return snapshot

    def _pop_group(self, key: Hashable, *, matched: bool) -> list[Code2WavAdmission]:
        pending = self._pending.pop(key, deque())
        if matched and len(pending) > 1:
            self._stats["matched_b2"] += 1
            self._stats["matched_requests"] += len(pending)
        elif not matched and pending:
            self._stats["deadline_b1"] += 1
            self._stats["deadline_requests"] += len(pending)
        if len(pending) > 1:
            skew_ms = (pending[-1].ready_at - pending[0].ready_at) * 1000.0
            self._stats["max_ready_skew_ms"] = max(self._stats["max_ready_skew_ms"], skew_ms)
        result: list[Code2WavAdmission] = []
        for item in pending:
            self._request_keys.pop(str(item.request.request_id), None)
            result.append(Code2WavAdmission(item.request, item.target_status))
        self._stats["pending"] = len(self._request_keys)
        return result
