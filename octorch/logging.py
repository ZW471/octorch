"""Console logging utilities for Octorch training and evaluation.

A port of Octax's logging module. The JAX-specific ``scan_with_progress`` /
``fori_loop_with_progress`` helpers (which used ``io_callback``) become plain
Python iteration helpers built on ``tqdm``.
"""

from __future__ import annotations

import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional

from tqdm import tqdm


class ConsoleLogger:
    """Flexible console logger with callback system and formatters."""

    def __init__(self, name: str = "Octorch", log_level: str = "INFO", use_colors: bool = True, show_timestamps: bool = True):
        self.name = name
        self.log_level = log_level.upper()
        self.use_colors = use_colors and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self.show_timestamps = show_timestamps
        self.start_time = time.time()
        colors = {"DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[35m", "RESET": "\033[0m"}
        self.colors = colors if self.use_colors else {k: "" for k in colors}
        self.level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

    def _should_log(self, level: str) -> bool:
        return self.level_order.get(level.upper(), 1) >= self.level_order.get(self.log_level, 1)

    def _format_message(self, level: str, message: str) -> str:
        timestamp = f"[{time.time() - self.start_time:8.2f}s]" if self.show_timestamps else ""
        level_str = f"[{level:>8s}]"
        if self.use_colors:
            level_str = f"{self.colors.get(level.upper(), '')}{level_str}{self.colors['RESET']}"
        return f"{timestamp}{level_str}[{self.name}] {message}"

    def log(self, level: str, message: str):
        if self._should_log(level):
            print(self._format_message(level.upper(), message))

    def debug(self, message: str): self.log("DEBUG", message)
    def info(self, message: str): self.log("INFO", message)
    def warning(self, message: str): self.log("WARNING", message)
    def error(self, message: str): self.log("ERROR", message)
    def critical(self, message: str): self.log("CRITICAL", message)


class TrainingLogger(ConsoleLogger):
    """Logger with training-specific helpers."""

    def __init__(self, name: str = "Training", **kwargs):
        super().__init__(name, **kwargs)
        self.step_count = 0

    def log_training_start(self, config: Dict[str, Any]):
        self.info("Training started")
        for k, v in config.items():
            self.info(f"  {k}: {v}")

    def log_training_step(self, step: int, metrics: Dict[str, Any], every: int = 1):
        self.step_count = step
        if step % every == 0:
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
            self.info(f"step {step}: " + ", ".join(parts))

    def log_training_end(self, final_metrics: Dict[str, Any]):
        self.info("Training finished")
        for k, v in final_metrics.items():
            self.info(f"  {k}: {v}")


class LoggingCallback:
    def on_training_start(self, config: Dict[str, Any]): ...
    def on_step(self, step: int, metrics: Dict[str, Any], state: Any = None): ...
    def on_training_end(self, final_metrics: Dict[str, Any]): ...


class ConsoleCallback(LoggingCallback):
    def __init__(self, log_interval: int = 10, logger: Optional[TrainingLogger] = None):
        self.log_interval = log_interval
        self.logger = logger or TrainingLogger()

    def on_training_start(self, config): self.logger.log_training_start(config)
    def on_step(self, step, metrics, state=None): self.logger.log_training_step(step, metrics, self.log_interval)
    def on_training_end(self, final_metrics): self.logger.log_training_end(final_metrics)


class MetricsCallback(LoggingCallback):
    def __init__(self, track_keys: Optional[List[str]] = None):
        self.track_keys = track_keys
        self.history: Dict[str, List[float]] = {}

    def on_step(self, step, metrics, state=None):
        for k, v in metrics.items():
            if self.track_keys is None or k in self.track_keys:
                self.history.setdefault(k, []).append(float(v))

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, values in self.history.items():
            n = len(values)
            mean = sum(values) / n
            out[k] = {"mean": mean, "min": min(values), "max": max(values), "last": values[-1],
                      "std": (sum((v - mean) ** 2 for v in values) / n) ** 0.5}
        return out


def with_logging(callbacks: Optional[List[LoggingCallback]] = None, config_key: str = "config"):
    """Decorator calling the callbacks around a ``train_fn(config, ...)``."""
    callbacks = callbacks or [ConsoleCallback()]

    def decorator(train_fn):
        @wraps(train_fn)
        def wrapper(*args, **kwargs):
            config = kwargs.get(config_key, args[0] if args else {})
            for cb in callbacks:
                cb.on_training_start(config if isinstance(config, dict) else {})
            result = train_fn(*args, **kwargs)
            for cb in callbacks:
                cb.on_training_end(result if isinstance(result, dict) else {"result": result})
            return result
        return wrapper
    return decorator


def progress(iterable: Iterable, total: Optional[int] = None, desc: str = "Running", **kwargs):
    """``tqdm`` wrapper used in place of Octax's ``scan_with_progress``."""
    return tqdm(iterable, total=total, desc=desc, **kwargs)


def loop_with_progress(fn: Callable[[Any, int], Any], carry: Any, length: int, desc: str = "Running", metrics_fn: Optional[Callable[[Any], Dict[str, Any]]] = None):
    """Run ``carry = fn(carry, i)`` for ``i in range(length)`` with a progress bar.

    Equivalent of ``fori_loop_with_progress`` / ``scan_with_progress_and_metrics``.
    """
    bar = tqdm(range(length), desc=desc)
    for i in bar:
        carry = fn(carry, i)
        if metrics_fn is not None:
            bar.set_postfix(metrics_fn(carry))
    return carry
