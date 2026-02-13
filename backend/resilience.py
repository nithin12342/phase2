"""
Resilience patterns for the Multimodal Deep Learning API.
Implements circuit breaker and retry with exponential backoff.
"""
import time
import logging
import functools
from typing import Callable, Type, Tuple, Optional
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is down.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, transitioning from OPEN to HALF_OPEN if timeout passed."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and \
                   (time.time() - self._last_failure_time) >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
            return self._state
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker CLOSED after successful recovery")
    
    def _record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap a function with circuit breaker logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except self.expected_exception as e:
                self._record_failure()
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except self.expected_exception as e:
                self._record_failure()
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


db_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=(Exception,)
)

ml_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60.0,
    expected_exception=(Exception,)
)

db_retry = retry_with_backoff(
    max_retries=3,
    base_delay=0.5,
    max_delay=10.0,
    exceptions=(Exception,)
)

vector_store_retry = retry_with_backoff(
    max_retries=2,
    base_delay=0.3,
    max_delay=5.0,
    exceptions=(Exception,)
)
