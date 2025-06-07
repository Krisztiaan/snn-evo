# keywords: [performance, async, compression, stats, profiling, optimized]
"""High-performance enhancement utilities for the export module."""

import threading
import queue
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import deque
import zlib
import lzma
import struct
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil


class AsyncWriteQueue:
    """High-performance asynchronous write queue with zero-copy operations."""
    
    def __init__(self, n_workers: int = 4, max_queue_size: int = 1000):
        """Initialize async write queue.
        
        Args:
            n_workers: Number of worker threads
            max_queue_size: Maximum items in queue before blocking
        """
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self._shutdown = False
        self._stats = {
            'writes': 0,
            'bytes_written': 0,
            'queue_wait_time': 0.0,
            'write_time': 0.0
        }
        self._stats_lock = threading.Lock()
        
        # Start worker threads
        for _ in range(n_workers):
            self.executor.submit(self._worker)
    
    def _worker(self):
        """Worker thread for processing write operations."""
        while not self._shutdown:
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:  # Shutdown signal
                    break
                    
                write_fn, args, kwargs = item
                start_time = time.perf_counter()
                
                # Execute write operation
                write_fn(*args, **kwargs)
                
                # Update stats
                write_time = time.perf_counter() - start_time
                with self._stats_lock:
                    self._stats['writes'] += 1
                    self._stats['write_time'] += write_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"AsyncWriteQueue error: {e}")
    
    def submit(self, write_fn: Callable, *args, **kwargs):
        """Submit a write operation to the queue.
        
        Args:
            write_fn: Function to call for writing
            *args: Positional arguments for write_fn
            **kwargs: Keyword arguments for write_fn
        """
        if self._shutdown:
            raise RuntimeError("AsyncWriteQueue is shutdown")
            
        start_time = time.perf_counter()
        self.queue.put((write_fn, args, kwargs))
        
        # Track queue wait time
        wait_time = time.perf_counter() - start_time
        with self._stats_lock:
            self._stats['queue_wait_time'] += wait_time
    
    def flush(self):
        """Wait for all pending writes to complete."""
        # Wait for queue to empty
        self.queue.join()
    
    def shutdown(self):
        """Shutdown the async write queue."""
        self._shutdown = True
        
        # Send shutdown signals
        for _ in range(len(self.workers)):
            self.queue.put(None)
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
            if stats['writes'] > 0:
                stats['avg_write_time'] = stats['write_time'] / stats['writes']
                stats['avg_queue_wait'] = stats['queue_wait_time'] / stats['writes']
            return stats


class AdaptiveCompressor:
    """Adaptive compression that selects optimal algorithm based on data characteristics."""
    
    def __init__(self):
        """Initialize adaptive compressor."""
        self.stats = {
            'gzip': {'count': 0, 'ratio': 0.0, 'time': 0.0},
            'lzf': {'count': 0, 'ratio': 0.0, 'time': 0.0},
            'lzma': {'count': 0, 'ratio': 0.0, 'time': 0.0},
            'none': {'count': 0, 'ratio': 1.0, 'time': 0.0}
        }
        self._lock = threading.Lock()
        
    def select_compression(self, data: np.ndarray, 
                         speed_priority: float = 0.5) -> Tuple[str, int]:
        """Select optimal compression based on data characteristics.
        
        Args:
            data: Data to analyze
            speed_priority: 0=max compression, 1=max speed
            
        Returns:
            Tuple of (algorithm, level)
        """
        # Fast path for small data
        if data.nbytes < 1024:  # Less than 1KB
            return 'none', 0
            
        # Analyze data characteristics
        data_flat = data.ravel()
        
        # Check sparsity
        sparsity = np.count_nonzero(data_flat) / data_flat.size
        
        # Check entropy (simplified)
        unique_ratio = len(np.unique(data_flat)) / data_flat.size
        
        # Decision logic
        if sparsity < 0.01:  # Very sparse
            return 'gzip', 1  # Fast compression for sparse data
        elif unique_ratio < 0.1:  # Low entropy
            if speed_priority > 0.7:
                return 'lzf', 0  # Fastest
            else:
                return 'gzip', 6  # Good compression
        elif unique_ratio > 0.9:  # High entropy
            if speed_priority > 0.5:
                return 'none', 0  # Skip compression
            else:
                return 'gzip', 1  # Light compression
        else:  # Medium entropy
            if speed_priority > 0.8:
                return 'lzf', 0
            elif speed_priority > 0.3:
                return 'gzip', 4
            else:
                return 'gzip', 9  # Max compression
    
    def update_stats(self, algorithm: str, original_size: int, 
                    compressed_size: int, compression_time: float):
        """Update compression statistics."""
        with self._lock:
            stats = self.stats[algorithm]
            stats['count'] += 1
            
            # Update rolling average
            ratio = compressed_size / original_size if original_size > 0 else 1.0
            alpha = 0.1  # Exponential moving average factor
            stats['ratio'] = (1 - alpha) * stats['ratio'] + alpha * ratio
            stats['time'] = (1 - alpha) * stats['time'] + alpha * compression_time
    
    def get_best_algorithm(self, speed_priority: float = 0.5) -> str:
        """Get best algorithm based on historical performance."""
        with self._lock:
            best_score = -1
            best_algo = 'gzip'
            
            for algo, stats in self.stats.items():
                if stats['count'] == 0:
                    continue
                    
                # Calculate score based on compression ratio and speed
                compression_score = 1.0 - stats['ratio']  # Lower ratio is better
                speed_score = 1.0 / (1.0 + stats['time'])  # Lower time is better
                
                score = (1 - speed_priority) * compression_score + speed_priority * speed_score
                
                if score > best_score:
                    best_score = score
                    best_algo = algo
                    
            return best_algo


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    count: int = 0
    total_time: float = 0.0
    total_bytes: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    def add_sample(self, time_sec: float, nbytes: int = 0):
        """Add a performance sample."""
        self.count += 1
        self.total_time += time_sec
        self.total_bytes += nbytes
        self.min_time = min(self.min_time, time_sec)
        self.max_time = max(self.max_time, time_sec)
    
    @property
    def avg_time(self) -> float:
        """Average time per operation."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def throughput_mbps(self) -> float:
        """Throughput in MB/s."""
        if self.total_time > 0:
            return (self.total_bytes / (1024 * 1024)) / self.total_time
        return 0.0


class RealtimeStats:
    """Real-time statistics tracking with minimal overhead."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize realtime stats tracker.
        
        Args:
            window_size: Size of sliding window for recent stats
        """
        self.window_size = window_size
        self.recent_times = deque(maxlen=window_size)
        self.recent_bytes = deque(maxlen=window_size)
        self.total_ops = 0
        self.total_bytes = 0
        self.start_time = time.perf_counter()
        self._lock = threading.Lock()
        
        # Memory tracking
        self.process = psutil.Process()
        self.peak_memory = 0
        
    def record_operation(self, operation_time: float, nbytes: int = 0):
        """Record an operation with minimal overhead."""
        with self._lock:
            self.recent_times.append(operation_time)
            self.recent_bytes.append(nbytes)
            self.total_ops += 1
            self.total_bytes += nbytes
            
            # Update peak memory periodically
            if self.total_ops % 100 == 0:
                current_memory = self.process.memory_info().rss
                self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.recent_times:
                return {
                    'total_ops': 0,
                    'throughput_mbps': 0.0,
                    'avg_op_time_ms': 0.0,
                    'peak_memory_mb': 0.0
                }
            
            recent_total_time = sum(self.recent_times)
            recent_total_bytes = sum(self.recent_bytes)
            elapsed_time = time.perf_counter() - self.start_time
            
            return {
                'total_ops': self.total_ops,
                'total_bytes_mb': self.total_bytes / (1024 * 1024),
                'elapsed_time_sec': elapsed_time,
                'overall_throughput_mbps': (self.total_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0,
                'recent_throughput_mbps': (recent_total_bytes / (1024 * 1024)) / recent_total_time if recent_total_time > 0 else 0,
                'avg_op_time_ms': (recent_total_time / len(self.recent_times)) * 1000,
                'peak_memory_mb': self.peak_memory / (1024 * 1024),
                'current_memory_mb': self.process.memory_info().rss / (1024 * 1024)
            }


class PerformanceProfiler:
    """Detailed performance profiler with minimal overhead."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        self.enabled = True
        
    def profile(self, operation: str, nbytes: int = 0) -> 'ProfileContext':
        """Create a profiling context for an operation."""
        return ProfileContext(self, operation, nbytes)
    
    def record(self, operation: str, time_sec: float, nbytes: int = 0):
        """Record a profiled operation."""
        if not self.enabled:
            return
            
        with self._lock:
            if operation not in self.metrics:
                self.metrics[operation] = PerformanceMetrics()
            self.metrics[operation].add_sample(time_sec, nbytes)
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        with self._lock:
            report = {}
            for op, metrics in self.metrics.items():
                report[op] = {
                    'count': metrics.count,
                    'total_time_sec': metrics.total_time,
                    'avg_time_ms': metrics.avg_time * 1000,
                    'min_time_ms': metrics.min_time * 1000,
                    'max_time_ms': metrics.max_time * 1000,
                    'throughput_mbps': metrics.throughput_mbps
                }
            return report
    
    def print_report(self):
        """Print formatted performance report."""
        report = self.get_report()
        
        print("\n=== Performance Report ===")
        print(f"{'Operation':<30} {'Count':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'MB/s':>10}")
        print("-" * 80)
        
        for op, stats in sorted(report.items()):
            print(f"{op:<30} {stats['count']:>10} {stats['avg_time_ms']:>10.2f} "
                  f"{stats['min_time_ms']:>10.2f} {stats['max_time_ms']:>10.2f} "
                  f"{stats['throughput_mbps']:>10.1f}")


class ProfileContext:
    """Context manager for profiling operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation: str, nbytes: int = 0):
        self.profiler = profiler
        self.operation = operation
        self.nbytes = nbytes
        self.start_time = None
        
    def __enter__(self):
        if self.profiler.enabled:
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler.enabled and self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.profiler.record(self.operation, elapsed, self.nbytes)