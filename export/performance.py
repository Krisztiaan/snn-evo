# keywords: [performance, async, compression, stats, profiling, optimized]
"""High-performance enhancement utilities for the export module."""

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import numpy as np
import psutil


class AsyncWriteQueue:
    """High-performance asynchronous write queue."""

    def __init__(self, n_workers: int = 4, max_queue_size: int = 1000):
        self.queue: queue.Queue[Optional[Tuple[Callable[..., Any], Tuple[Any, ...]]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="AsyncWrite")
        self._stats: Dict[str, Any] = {"writes": 0, "bytes_written": 0, "write_time": 0.0}
        self._stats_lock = threading.Lock()

    def submit(self, write_fn: Callable[..., Any], *args: Any) -> None:
        self.queue.put((write_fn, args))

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                break
            write_fn, args = item
            start_time = time.perf_counter()
            try:
                write_fn(*args)
            except Exception as e:
                print(f"AsyncWriteQueue error: {e}")
            finally:
                write_time = time.perf_counter() - start_time
                with self._stats_lock:
                    self._stats["writes"] += 1
                    self._stats["write_time"] += write_time
                self.queue.task_done()

    def start_workers(self) -> None:
        for _ in range(self.executor._max_workers):
            self.executor.submit(self._worker)

    def shutdown(self, wait: bool = True) -> None:
        for _ in range(self.executor._max_workers):
            self.queue.put(None)
        self.executor.shutdown(wait=wait)

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = self._stats.copy()
            if stats["writes"] > 0:
                stats["avg_write_time_ms"] = (stats["write_time"] / stats["writes"]) * 1000
            return stats


class AdaptiveCompressor:
    """Selects optimal compression algorithm based on data characteristics."""

    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float]] = {
            "gzip": {"count": 0, "ratio": 0.0, "time": 0.0},
            "lzf": {"count": 0, "ratio": 0.0, "time": 0.0},
            "none": {"count": 0, "ratio": 1.0, "time": 0.0},
        }

    def select_compression(
        self, data: np.ndarray, speed_priority: float = 0.5
    ) -> Tuple[Optional[str], int]:
        if data.nbytes < 1024:
            return None, 0

        unique_ratio = len(np.unique(data.ravel())) / data.size
        if unique_ratio < 0.1:  # Low entropy, highly compressible
            return "gzip", 6
        elif unique_ratio > 0.9:  # High entropy, less compressible
            return "lzf" if speed_priority > 0.5 else "gzip", 1
        else:  # Medium entropy
            return "gzip", 4


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    def add_sample(self, time_sec: float) -> None:
        self.count += 1
        self.total_time += time_sec
        self.min_time = min(self.min_time, time_sec)
        self.max_time = max(self.max_time, time_sec)

    @property
    def avg_time_ms(self) -> float:
        return (self.total_time / self.count) * 1000 if self.count > 0 else 0.0


class RealtimeStats:
    """Real-time statistics tracking with minimal overhead."""

    def __init__(self, window_size: int = 1000):
        self.recent_times: Deque[float] = deque(maxlen=window_size)
        self.total_ops = 0
        self.start_time = time.perf_counter()
        self.process = psutil.Process()
        self.peak_memory = 0

    def record_operation(self, operation_time: float) -> None:
        self.recent_times.append(operation_time)
        self.total_ops += 1
        if self.total_ops % 100 == 0:
            self.peak_memory = max(self.peak_memory, self.process.memory_info().rss)

    def get_summary(self) -> Dict[str, Any]:
        if not self.recent_times:
            return {}
        recent_total_time = sum(self.recent_times)
        return {
            "total_ops": self.total_ops,
            "avg_op_time_ms": (recent_total_time / len(self.recent_times)) * 1000,
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
        }


class PerformanceProfiler:
    """Detailed performance profiler with context manager."""

    def __init__(self) -> None:
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._lock = threading.Lock()

    def profile(self, operation: str) -> "ProfileContext":
        return ProfileContext(self, operation)

    def record(self, operation: str, time_sec: float) -> None:
        with self._lock:
            self.metrics[operation].add_sample(time_sec)

    def get_report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                op: {
                    "count": m.count,
                    "avg_time_ms": m.avg_time_ms,
                    "min_time_ms": m.min_time * 1000,
                    "max_time_ms": m.max_time * 1000,
                }
                for op, m in self.metrics.items()
            }

    def print_report(self) -> None:
        report = self.get_report()
        print("\n=== Performance Report ===")
        print(f"{'Operation':<30} {'Count':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("-" * 75)
        for op, stats in sorted(report.items()):
            print(
                f"{op:<30} {stats['count']:>10} {stats['avg_time_ms']:>10.2f} "
                f"{stats['min_time_ms']:>10.2f} {stats['max_time_ms']:>10.2f}"
            )


class ProfileContext:
    """Context manager for profiling operations."""

    def __init__(self, profiler: PerformanceProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
        self.start_time: Optional[float] = None

    def __enter__(self) -> "ProfileContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.profiler.record(self.operation, elapsed)
