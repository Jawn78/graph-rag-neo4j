"""
Centralized logging configuration for Graph RAG.

Provides structured logging with trace IDs for request correlation.
"""

import logging
import uuid
import contextvars
from functools import wraps
from typing import Optional, Callable, Any
import time

# Context variable for trace ID (thread-safe)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar('trace_id', default='')


def get_trace_id() -> str:
    """Get current trace ID or generate a new one."""
    tid = _trace_id.get()
    if not tid:
        tid = uuid.uuid4().hex[:8]
        _trace_id.set(tid)
    return tid


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """Set trace ID for current context. Returns the trace ID."""
    tid = trace_id or uuid.uuid4().hex[:8]
    _trace_id.set(tid)
    return tid


def clear_trace_id() -> None:
    """Clear the trace ID for current context."""
    _trace_id.set('')


class TraceFormatter(logging.Formatter):
    """Formatter that includes trace ID in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = _trace_id.get()
        if trace_id:
            record.trace_id = f"[{trace_id}]"
        else:
            record.trace_id = ""
        return super().format(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    handler = logging.StreamHandler()
    handler.setFormatter(TraceFormatter(
        '%(asctime)s %(levelname)s %(trace_id)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Quiet noisy libraries
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)


def timed(operation_name: str) -> Callable:
    """
    Decorator to log operation timing.

    Usage:
        @timed("embedding")
        def embed_chunks(chunks):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)
            trace_id = get_trace_id()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(f"[{trace_id}] {operation_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"[{trace_id}] {operation_name} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
