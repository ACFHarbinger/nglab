from unittest.mock import AsyncMock, patch

import pytest

from python.src.db.cache import (
    cache_key,
    cached_query,
    get_cache_stats,
    get_redis_client,
    invalidate_cache,
)


class TestCache:
    def test_cache_key_generation(self):
        # Test that same args produce same key
        key1 = cache_key("arg1", "arg2", kwarg1="value1")
        key2 = cache_key("arg1", "arg2", kwarg1="value1")
        assert key1 == key2

        # Test that different args produce different keys
        key3 = cache_key("arg1", "arg3", kwarg1="value1")
        assert key1 != key3

    @pytest.mark.anyio
    async def test_get_redis_client(self):
        with patch("python.src.db.cache.redis.Redis") as mock_redis_class:
            mock_client = AsyncMock()
            mock_redis_class.return_value = mock_client

            # Reset global client to test initialization
            import python.src.db.cache as cache_module

            cache_module._redis_client = None

            client = await get_redis_client()

            assert client is not None
            mock_redis_class.assert_called_once()

    @pytest.mark.anyio
    async def test_cached_query_hit(self):
        mock_client = AsyncMock()
        mock_client.get.return_value = '{"result": "cached_data"}'

        with patch("python.src.db.cache.get_redis_client", return_value=mock_client):

            @cached_query(ttl=300, key_prefix="test")
            async def test_func(arg1):
                return {"result": "fresh_data"}

            result = await test_func("value1")

            assert result == {"result": "cached_data"}
            mock_client.get.assert_called_once()

    @pytest.mark.anyio
    async def test_cached_query_miss(self):
        mock_client = AsyncMock()
        mock_client.get.return_value = None

        with patch("python.src.db.cache.get_redis_client", return_value=mock_client):

            @cached_query(ttl=300, key_prefix="test")
            async def test_func(arg1):
                return {"result": "fresh_data"}

            result = await test_func("value1")

            assert result == {"result": "fresh_data"}
            mock_client.get.assert_called_once()
            mock_client.setex.assert_called_once()

    @pytest.mark.anyio
    async def test_cached_query_error_handling(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Redis error")

        with patch("python.src.db.cache.get_redis_client", return_value=mock_client):

            @cached_query(ttl=300, key_prefix="test")
            async def test_func(arg1):
                return {"result": "fresh_data"}

            result = await test_func("value1")

            # Should still return fresh data even if cache fails
            assert result == {"result": "fresh_data"}

    @pytest.mark.anyio
    async def test_invalidate_cache(self):
        mock_client = AsyncMock()
        mock_client.keys.return_value = ["key1", "key2", "key3"]
        mock_client.delete.return_value = 3

        with patch("python.src.db.cache.get_redis_client", return_value=mock_client):
            count = await invalidate_cache("test_*")

            assert count == 3
            mock_client.keys.assert_called_once_with("test_*")
            mock_client.delete.assert_called_once_with("key1", "key2", "key3")

    @pytest.mark.anyio
    async def test_invalidate_cache_no_keys(self):
        mock_client = AsyncMock()
        mock_client.keys.return_value = []

        with patch("python.src.db.cache.get_redis_client", return_value=mock_client):
            count = await invalidate_cache("nonexistent_*")

            assert count == 0
            mock_client.delete.assert_not_called()

    @pytest.mark.anyio
    async def test_get_cache_stats(self):
        mock_client = AsyncMock()
        mock_client.info.return_value = {"keyspace_hits": 100, "keyspace_misses": 50}

        with patch("python.src.db.cache.get_redis_client", return_value=mock_client):
            stats = await get_cache_stats()

            assert stats["keyspace_hits"] == 100
            assert stats["keyspace_misses"] == 50
            assert stats["hit_rate"] == pytest.approx(100 / 150)
