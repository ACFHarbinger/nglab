import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from python.src.db.cache import (
    get_redis_client,
    cache_key,
    cached_query,
    invalidate_cache,
    get_cache_stats
)

class TestCache(unittest.TestCase):
    def test_cache_key_generation(self):
        # Test that same args produce same key
        key1 = cache_key("arg1", "arg2", kwarg1="value1")
        key2 = cache_key("arg1", "arg2", kwarg1="value1")
        self.assertEqual(key1, key2)
        
        # Test that different args produce different keys
        key3 = cache_key("arg1", "arg3", kwarg1="value1")
        self.assertNotEqual(key1, key3)
    
    @pytest.mark.asyncio
    async def test_get_redis_client(self):
        with patch('python.src.db.cache.redis.Redis') as mock_redis_class:
            mock_client = AsyncMock()
            mock_redis_class.return_value = mock_client
            
            # Reset global client to test initialization
            import python.src.db.cache as cache_module
            cache_module._redis_client = None
            
            client = await get_redis_client()
            
            self.assertIsNotNone(client)
            mock_redis_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_query_hit(self):
        mock_client = AsyncMock()
        mock_client.get.return_value = '{"result": "cached_data"}'
        
        with patch('python.src.db.cache.get_redis_client', return_value=mock_client):
            @cached_query(ttl=300, key_prefix="test")
            async def test_func(arg1):
                return {"result": "fresh_data"}
            
            result = await test_func("value1")
            
            self.assertEqual(result, {"result": "cached_data"})
            mock_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_query_miss(self):
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        
        with patch('python.src.db.cache.get_redis_client', return_value=mock_client):
            @cached_query(ttl=300, key_prefix="test")
            async def test_func(arg1):
                return {"result": "fresh_data"}
            
            result = await test_func("value1")
            
            self.assertEqual(result, {"result": "fresh_data"})
            mock_client.get.assert_called_once()
            mock_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_query_error_handling(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Redis error")
        
        with patch('python.src.db.cache.get_redis_client', return_value=mock_client):
            @cached_query(ttl=300, key_prefix="test")
            async def test_func(arg1):
                return {"result": "fresh_data"}
            
            result = await test_func("value1")
            
            # Should still return fresh data even if cache fails
            self.assertEqual(result, {"result": "fresh_data"})
    
    @pytest.mark.asyncio
    async def test_invalidate_cache(self):
        mock_client = AsyncMock()
        mock_client.keys.return_value = ["key1", "key2", "key3"]
        mock_client.delete.return_value = 3
        
        with patch('python.src.db.cache.get_redis_client', return_value=mock_client):
            count = await invalidate_cache("test_*")
            
            self.assertEqual(count, 3)
            mock_client.keys.assert_called_once_with("test_*")
            mock_client.delete.assert_called_once_with("key1", "key2", "key3")
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_no_keys(self):
        mock_client = AsyncMock()
        mock_client.keys.return_value = []
        
        with patch('python.src.db.cache.get_redis_client', return_value=mock_client):
            count = await invalidate_cache("nonexistent_*")
            
            self.assertEqual(count, 0)
            mock_client.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        mock_client = AsyncMock()
        mock_client.info.return_value = {
            "keyspace_hits": 100,
            "keyspace_misses": 50
        }
        
        with patch('python.src.db.cache.get_redis_client', return_value=mock_client):
            stats = await get_cache_stats()
            
            self.assertEqual(stats["keyspace_hits"], 100)
            self.assertEqual(stats["keyspace_misses"], 50)
            self.assertAlmostEqual(stats["hit_rate"], 100 / 150)
