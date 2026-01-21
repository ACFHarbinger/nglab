"""
Tests for cache utilities.
"""

from python.src.db.cache import cache_key


class TestCacheKey:
    """Tests for the cache_key function."""

    def test_cache_key_simple_args(self):
        """Test cache key generation with simple arguments."""
        key = cache_key(1, 2, 3)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

    def test_cache_key_with_kwargs(self):
        """Test cache key generation with kwargs."""
        key = cache_key(user_id=123, model="gpt-4")

        assert isinstance(key, str)
        assert len(key) == 64

    def test_cache_key_deterministic(self):
        """Test that same inputs produce same key."""
        key1 = cache_key(1, 2, 3, name="test")
        key2 = cache_key(1, 2, 3, name="test")

        assert key1 == key2

    def test_cache_key_different_args_different_key(self):
        """Test that different args produce different keys."""
        key1 = cache_key(1, 2, 3)
        key2 = cache_key(1, 2, 4)

        assert key1 != key2

    def test_cache_key_different_kwargs_different_key(self):
        """Test that different kwargs produce different keys."""
        key1 = cache_key(name="test")
        key2 = cache_key(name="other")

        assert key1 != key2

    def test_cache_key_order_independent_kwargs(self):
        """Test that kwargs order doesn't matter (sorted)."""
        key1 = cache_key(a=1, b=2, c=3)
        key2 = cache_key(c=3, a=1, b=2)

        assert key1 == key2

    def test_cache_key_empty(self):
        """Test cache key with no arguments."""
        key = cache_key()

        assert isinstance(key, str)
        assert len(key) == 64

    def test_cache_key_mixed(self):
        """Test cache key with both args and kwargs."""
        key1 = cache_key(1, "hello", flag=True, count=10)
        key2 = cache_key(1, "hello", count=10, flag=True)

        assert key1 == key2

    def test_cache_key_with_list(self):
        """Test cache key with list argument."""
        key = cache_key([1, 2, 3])

        assert isinstance(key, str)
        assert len(key) == 64

    def test_cache_key_with_dict(self):
        """Test cache key with dict argument."""
        key = cache_key({"a": 1, "b": 2})

        assert isinstance(key, str)
        assert len(key) == 64

    def test_cache_key_nested_structures(self):
        """Test cache key with nested data structures."""
        data = {
            "users": [1, 2, 3],
            "config": {"batch_size": 32, "lr": 0.001},
        }
        key = cache_key(data)

        assert isinstance(key, str)
        assert len(key) == 64

        # Same nested structure should produce same key
        key2 = cache_key(data)
        assert key == key2
