"""
Middleware module for the Multimodal Deep Learning API.
Contains rate limiting, request ID, and timeout middleware.
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import asyncio
from config import get_settings

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    Limits requests per IP address.
    """
    
    def __init__(self, app, requests_limit: int = None, window_seconds: int = None):
        super().__init__(app)
        self.requests_limit = requests_limit or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window_seconds
        self.request_counts = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Clean up stale IPs every hour
    
    def _cleanup_stale_ips(self, current_time: float):
        """Remove IPs that haven't made requests in the last cleanup interval."""
        if current_time - self._last_cleanup > self._cleanup_interval:
            stale_ips = [
                ip for ip, timestamps in self.request_counts.items()
                if not timestamps or max(timestamps) < current_time - self._cleanup_interval
            ]
            for ip in stale_ips:
                del self.request_counts[ip]
            self._last_cleanup = current_time
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        
        if request.url.path in ["/health", "/ready", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        self._cleanup_stale_ips(current_time)
        
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > window_start
        ]
        
        if len(self.request_counts[client_ip]) >= self.requests_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.requests_limit} requests per {self.window_seconds} seconds."
            )
        
        self.request_counts[client_ip].append(current_time)
        
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_limit - len(self.request_counts[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(window_start + self.window_seconds))
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Adds a unique request ID to each request for tracing.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Adds timeout to requests to prevent long-running operations.
    """
    
    def __init__(self, app, timeout_seconds: int = None):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds or settings.request_timeout_seconds
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            return response
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Request timeout after {self.timeout_seconds} seconds"
            )
