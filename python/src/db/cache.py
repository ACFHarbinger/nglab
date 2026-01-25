"""
Redis-based query result caching layer for NGLab.

Provides decorators and utilities for caching expensive database queries
with automatic invalidation on writes.
"""

import functools
import hashlib
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, cast

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Type variable for generic function return types
T = TypeVar("T")

# Global Redis client (initialized on first use)
_redis_client: redis.Redis | None = None


async def get_redis_client() -> redis.Redis:
    """Get or create singleton Redis client."""
    global _redis_client  # noqa: PLW0603
    if _redis_client is None:
        _redis_client = redis.Redis(
            host="redis",
            port=6379,
            db=0,
            decode_responses=True,
        )
    return _redis_client


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from function arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


def cached_query(
    ttl: int = 300, key_prefix: str = "query"
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    """
    Decorator for caching query results in Redis.

    Args:
        ttl: Time-to-live in seconds (default 5 minutes)
        key_prefix: Prefix for cache keys

    Example:
        @cached_query(ttl=600, key_prefix="user_models")
        async def get_user_models(user_id: int):
            return await db.fetch_all(...)
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        """Inner decorator function."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that handles Redis lookups and cache population."""
            client = await get_redis_client()

            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            try:
                cached = await client.get(key)
                if cached:
                    logger.debug(f"Cache hit: {key}")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await client.setex(
                    key,
                    ttl,
                    json.dumps(
                        result, default=str
                    ),  # default=str for datetime serialization
                )
                logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

            return result

        return cast(Callable[..., Coroutine[Any, Any, T]], wrapper)

    return decorator


async def invalidate_cache(pattern: str) -> int:
    """
    Invalidate cache keys matching pattern.

    Args:
        pattern: Redis key pattern (e.g., "user_models:*")

    Returns:
        Number of keys invalidated
    """
    client = await get_redis_client()
    keys = await client.keys(pattern)
    if keys:
        count = cast(int, await client.delete(*keys))
        logger.info(f"Invalidated {count} cache keys matching '{pattern}'")
        return count
    return 0


async def get_cache_stats() -> dict[str, Any]:
    """Get Redis cache statistics."""
    client = await get_redis_client()
    info = await client.info("stats")
    return {
        "keyspace_hits": info.get("keyspace_hits", 0),
        "keyspace_misses": info.get("keyspace_misses", 0),
        "hit_rate": (
            info.get("keyspace_hits", 0)
            / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
        ),
    }
