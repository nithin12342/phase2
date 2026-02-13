"""
Caching module for the Multimodal Deep Learning API.
Provides Redis caching with fallback to in-memory cache.
"""
import hashlib
import json
import time
import logging
from typing import Optional, Any, Dict
from functools import wraps
import threading

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class InMemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry["expires_at"] is None or entry["expires_at"] > time.time():
                    return entry["value"]
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL in seconds."""
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].get("created_at", 0))
                del self._cache[oldest_key]
            
            self._cache[key] = {
                "value": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl if ttl else None
            }
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "backend": "in_memory"
            }


class RedisCache:
    """Redis-based cache with connection handling."""
    
    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            import redis
            self._client = redis.from_url(self._redis_url)
            self._client.ping()
            logger.info("Connected to Redis cache")
        except ImportError:
            logger.warning("Redis package not installed. Install with: pip install redis")
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.is_connected:
            return None
        try:
            value = self._client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis with optional TTL."""
        if not self.is_connected:
            return
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                self._client.setex(key, ttl, serialized)
            else:
                self._client.set(key, serialized)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.is_connected:
            return False
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries (use with caution!)."""
        if not self.is_connected:
            return
        try:
            self._client.flushdb()
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        if not self.is_connected:
            return {"backend": "redis", "connected": False}
        try:
            info = self._client.info()
            return {
                "backend": "redis",
                "connected": True,
                "used_memory": info.get("used_memory_human"),
                "keys": self._client.dbsize()
            }
        except Exception:
            return {"backend": "redis", "connected": False}


class CacheManager:
    """
    Unified cache manager that uses Redis if available, 
    falls back to in-memory cache.
    """
    
    def __init__(self):
        self._redis_cache: Optional[RedisCache] = None
        self._memory_cache = InMemoryCache(max_size=1000)
        
        if settings.redis_url:
            self._redis_cache = RedisCache(settings.redis_url)
    
    @property
    def _active_cache(self):
        """Get the active cache backend."""
        if self._redis_cache and self._redis_cache.is_connected:
            return self._redis_cache
        return self._memory_cache
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._active_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = 3600):
        """Set value in cache with TTL (default 1 hour)."""
        self._active_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self._active_cache.delete(key)
    
    def clear(self):
        """Clear cache."""
        self._active_cache.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return self._active_cache.stats()


cache = CacheManager()


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (default 1 hour)
        key_prefix: Prefix for cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


def cache_embedding(text_hash: str, embedding: list, ttl: int = 86400):
    """Cache an embedding with 24-hour default TTL."""
    cache.set(f"embedding:{text_hash}", embedding, ttl)


def get_cached_embedding(text_hash: str) -> Optional[list]:
    """Get a cached embedding."""
    return cache.get(f"embedding:{text_hash}")
