"""
Performance monitoring utilities for the graph RAG system.
"""
import time
import psutil
import os
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager

class PerformanceMonitor:
    """Monitor system and application performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        return duration
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        process = psutil.Process(os.getpid())
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "process_cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "threads": process.num_threads()
        }
    
    def get_operation_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        
        times = self.metrics[operation]
        return {
            "count": len(times),
            "total": sum(times),
            "average": sum(times) / len(times),
            "min": min(times),
            "max": max(times)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics."""
        stats = {
            "system": self.get_system_metrics(),
            "operations": {}
        }
        
        for operation in self.metrics:
            stats["operations"][operation] = self.get_operation_stats(operation)
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()

# Global monitor instance
monitor = PerformanceMonitor()

def timed_operation(operation_name: str):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(operation_name)
                print(f"[PERF] {operation_name}: {duration:.3f}s")
        return wrapper
    return decorator

@contextmanager
def timed_context(operation_name: str):
    """Context manager for timing code blocks."""
    monitor.start_timer(operation_name)
    try:
        yield
    finally:
        duration = monitor.end_timer(operation_name)
        print(f"[PERF] {operation_name}: {duration:.3f}s")

def log_performance_summary():
    """Log a summary of all performance metrics."""
    stats = monitor.get_all_stats()
    print("\n=== PERFORMANCE SUMMARY ===")
    
    # System metrics
    sys_metrics = stats["system"]
    print(f"CPU: {sys_metrics['cpu_percent']:.1f}%")
    print(f"Memory: {sys_metrics['memory_percent']:.1f}%")
    print(f"Process Memory: {sys_metrics['process_memory_mb']:.1f} MB")
    print(f"Threads: {sys_metrics['threads']}")
    
    # Operation metrics
    print("\nOperation Times:")
    for op, op_stats in stats["operations"].items():
        if op_stats:
            print(f"  {op}: {op_stats['count']} calls, "
                  f"avg: {op_stats['average']:.3f}s, "
                  f"total: {op_stats['total']:.3f}s")
    
    print("=" * 30)
