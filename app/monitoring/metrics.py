"""System monitoring and metrics collection."""

import time
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    start_time: datetime
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque  # Keep last N response times

    # Endpoint specific metrics
    endpoint_metrics: dict = None

    def __post_init__(self):
        """Initialize endpoint metrics."""
        if self.endpoint_metrics is None:
            self.endpoint_metrics = {}
        # Keep last 1000 response times for rolling average
        self.response_times = deque(maxlen=1000)

    @property
    def uptime_seconds(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def avg_response_time_ms(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    @property
    def success_rate(self) -> float:
        """Get request success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    def record_request(self, response_time_ms: float, endpoint: str, success: bool = True):
        """Record a request."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

        self.response_times.append(response_time_ms)

        # Track per-endpoint metrics
        if endpoint not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint] = {
                "requests": 0,
                "failures": 0,
                "response_times": deque(maxlen=100)
            }

        self.endpoint_metrics[endpoint]["requests"] += 1
        if not success:
            self.endpoint_metrics[endpoint]["failures"] += 1
        self.endpoint_metrics[endpoint]["response_times"].append(response_time_ms)

    def get_endpoint_stats(self, endpoint: str) -> dict:
        """Get statistics for a specific endpoint."""
        if endpoint not in self.endpoint_metrics:
            return {}

        stats = self.endpoint_metrics[endpoint]
        response_times = stats["response_times"]

        return {
            "endpoint": endpoint,
            "total_requests": stats["requests"],
            "failures": stats["failures"],
            "success_rate": (stats["requests"] - stats["failures"]) / stats["requests"],
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
        }

    def get_all_stats(self) -> dict:
        """Get comprehensive system statistics."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "endpoints": {
                endpoint: self.get_endpoint_stats(endpoint)
                for endpoint in self.endpoint_metrics
            }
        }


class MetricsCollector:
    """Collect and manage system metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = SystemMetrics(start_time=datetime.utcnow())

    def start_timer(self) -> float:
        """Start a timer. Return start time."""
        return time.time()

    def end_timer(self, start_time: float) -> float:
        """End a timer. Return elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000

    def record_request(
        self,
        response_time_ms: float,
        endpoint: str,
        success: bool = True
    ):
        """Record a request metric."""
        self.metrics.record_request(response_time_ms, endpoint, success)

    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        return self.metrics.get_all_stats()

    def get_endpoint_metrics(self, endpoint: str) -> dict:
        """Get metrics for specific endpoint."""
        return self.metrics.get_endpoint_stats(endpoint)

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = SystemMetrics(start_time=datetime.utcnow())


# Global metrics instance
metrics_collector = MetricsCollector()
