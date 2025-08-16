"""Test suite for the Cache module."""
import pytest
from unittest.mock import MagicMock, patch, mock_open, Mock
from pathlib import Path
import json
import sqlite3
import shutil
import os
import pickle
import hashlib
from datetime import datetime

from ultrathink_cli.cache import CacheManager


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for cache testing."""
    return tmp_path / "test_cache"


@pytest.fixture
def cache_manager(temp_dir):
    """Create a CacheManager instance with a temporary directory."""
    return CacheManager(base_dir=temp_dir)


@pytest.fixture
def sample_kernel():
    """Create a sample kernel data structure."""
    return {
        "code": "__global__ void test_kernel() { /* kernel code */ }",
        "operation": "matmul",
        "constraints": {"max_registers": 255, "shared_memory_kb": 48},
        "benchmark": {"avg_time_ms": 10.5, "baseline_time_ms": 20.0},
        "profile": {"memory_usage_mb": 256.0}
    }


@pytest.fixture
def mock_sqlite_connect():
    """Mock sqlite3.connect."""
    with patch('ultrathink_cli.cache.sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_connect.return_value = mock_conn
        yield mock_connect, mock_conn, mock_cursor


class TestCacheManagerInitialization:
    """Test CacheManager initialization."""

    def test_initialization_default_path(self):
        """Test initialization with default path."""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/home/user")
            with patch('pathlib.Path.mkdir'):
                with patch('ultrathink_cli.cache.sqlite3.connect'):
                    cm = CacheManager()
                    assert cm.base_dir == Path("/home/user/.ultrathink-cli")

    def test_initialization_custom_path(self, temp_dir):
        """Test initialization with custom path."""
        cm = CacheManager(base_dir=temp_dir)
        assert cm.base_dir == temp_dir
        assert cm.config_dir == temp_dir
        assert cm.cache_dir == temp_dir / "cache"
        assert cm.kernels_dir == temp_dir / "cache" / "kernels"
        assert cm.db_path == temp_dir / "cache" / "metadata.db"

    def test_directory_creation(self, temp_dir):
        """Test that directories are created during initialization."""
        cm = CacheManager(base_dir=temp_dir)
        assert cm.config_dir.exists()
        assert cm.cache_dir.exists()
        assert cm.kernels_dir.exists()

    def test_database_initialization(self, temp_dir):
        """Test that database is initialized with correct schema."""
        cm = CacheManager(base_dir=temp_dir)
        
        # Check that database file exists
        assert cm.db_path.exists()
        
        # Verify tables were created
        with sqlite3.connect(cm.db_path) as conn:
            cursor = conn.cursor()
            
            # Check kernel_metadata table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kernel_metadata'")
            assert cursor.fetchone() is not None
            
            # Check optimization_history table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='optimization_history'")
            assert cursor.fetchone() is not None


class TestConfigManagement:
    """Test configuration management methods."""

    def test_get_config_path(self, cache_manager):
        """Test getting config file path."""
        config_path = cache_manager.get_config_path()
        assert config_path == cache_manager.config_dir / "config.json"

    def test_get_config_empty(self, cache_manager):
        """Test getting config when file doesn't exist."""
        config = cache_manager.get_config()
        assert config == {}

    def test_get_config_with_data(self, cache_manager):
        """Test getting config with existing data."""
        config_data = {"openrouter_api_key": "test-key", "other_setting": "value"}
        config_path = cache_manager.get_config_path()
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = cache_manager.get_config()
        assert config == config_data

    def test_save_config(self, cache_manager):
        """Test saving configuration."""
        config_data = {"openrouter_api_key": "new-key", "setting": "value"}
        cache_manager.save_config(config_data)
        
        # Verify file was written
        config_path = cache_manager.get_config_path()
        assert config_path.exists()
        
        # Verify content
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config == config_data

    def test_has_api_key_false(self, cache_manager):
        """Test has_api_key when no key is configured."""
        assert cache_manager.has_api_key() is False

    def test_has_api_key_true(self, cache_manager):
        """Test has_api_key when key is configured."""
        cache_manager.save_config({"openrouter_api_key": "test-key"})
        assert cache_manager.has_api_key() is True

    def test_get_api_key_none(self, cache_manager):
        """Test get_api_key when no key is configured."""
        assert cache_manager.get_api_key() is None

    def test_get_api_key_exists(self, cache_manager):
        """Test get_api_key when key exists."""
        cache_manager.save_config({"openrouter_api_key": "test-api-key-123"})
        assert cache_manager.get_api_key() == "test-api-key-123"

    def test_save_api_key(self, cache_manager):
        """Test saving API key."""
        cache_manager.save_api_key("new-api-key")
        
        config = cache_manager.get_config()
        assert config["openrouter_api_key"] == "new-api-key"


class TestKernelCaching:
    """Test kernel caching functionality."""

    def test_generate_kernel_id(self, cache_manager):
        """Test kernel ID generation."""
        kernel_id = cache_manager._generate_kernel_id("model1", "matmul", "code1")
        assert isinstance(kernel_id, str)
        assert len(kernel_id) == 16
        
        # Same inputs should generate same ID
        kernel_id2 = cache_manager._generate_kernel_id("model1", "matmul", "code1")
        assert kernel_id == kernel_id2
        
        # Different inputs should generate different IDs
        kernel_id3 = cache_manager._generate_kernel_id("model2", "matmul", "code1")
        assert kernel_id != kernel_id3

    def test_cache_kernel(self, cache_manager, sample_kernel):
        """Test caching a kernel."""
        kernel_id = cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        
        assert isinstance(kernel_id, str)
        
        # Verify kernel file was created
        kernel_file = cache_manager.kernels_dir / f"{kernel_id}.pkl"
        assert kernel_file.exists()
        
        # Verify database entry
        with sqlite3.connect(cache_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kernel_metadata WHERE id = ?", (kernel_id,))
            row = cursor.fetchone()
            assert row is not None

    def test_get_cached_kernel_exists(self, cache_manager, sample_kernel):
        """Test retrieving a cached kernel."""
        # Cache a kernel first
        cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        
        # Retrieve it
        cached_kernel = cache_manager.get_cached_kernel("llama-7b", "matmul")
        assert cached_kernel is not None
        assert cached_kernel["code"] == sample_kernel["code"]
        assert cached_kernel["operation"] == sample_kernel["operation"]

    def test_get_cached_kernel_not_exists(self, cache_manager):
        """Test retrieving non-existent kernel."""
        cached_kernel = cache_manager.get_cached_kernel("non-existent", "matmul")
        assert cached_kernel is None

    def test_get_cached_kernel_updates_last_used(self, cache_manager, sample_kernel):
        """Test that retrieving a kernel updates last_used timestamp."""
        kernel_id = cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        
        # Get initial last_used
        with sqlite3.connect(cache_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_used FROM kernel_metadata WHERE id = ?", (kernel_id,))
            initial_last_used = cursor.fetchone()[0]
        
        # Wait a bit and retrieve kernel
        import time
        time.sleep(0.1)
        cache_manager.get_cached_kernel("llama-7b", "matmul")
        
        # Check last_used was updated
        with sqlite3.connect(cache_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_used FROM kernel_metadata WHERE id = ?", (kernel_id,))
            new_last_used = cursor.fetchone()[0]
        
        assert new_last_used > initial_last_used

    def test_list_cached_kernels_empty(self, cache_manager):
        """Test listing kernels when none are cached."""
        kernels = cache_manager.list_cached_kernels()
        assert kernels == []

    def test_list_cached_kernels_all(self, cache_manager, sample_kernel):
        """Test listing all cached kernels."""
        # Cache multiple kernels
        cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        cache_manager.cache_kernel("mistral-7b", "attention", sample_kernel)
        
        kernels = cache_manager.list_cached_kernels()
        assert len(kernels) == 2
        assert any(k["model"] == "llama-7b" and k["operation"] == "matmul" for k in kernels)
        assert any(k["model"] == "mistral-7b" and k["operation"] == "attention" for k in kernels)

    def test_list_cached_kernels_filtered(self, cache_manager, sample_kernel):
        """Test listing kernels filtered by model."""
        # Cache multiple kernels
        cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        cache_manager.cache_kernel("llama-7b", "attention", sample_kernel)
        cache_manager.cache_kernel("mistral-7b", "matmul", sample_kernel)
        
        kernels = cache_manager.list_cached_kernels("llama-7b")
        assert len(kernels) == 2
        assert all(k["model"] == "llama-7b" for k in kernels)

    def test_count_cached_kernels(self, cache_manager, sample_kernel):
        """Test counting cached kernels."""
        assert cache_manager.count_cached_kernels() == 0
        
        cache_manager.cache_kernel("model1", "op1", sample_kernel)
        assert cache_manager.count_cached_kernels() == 1
        
        cache_manager.cache_kernel("model2", "op2", sample_kernel)
        assert cache_manager.count_cached_kernels() == 2

    def test_clear_cache(self, cache_manager, sample_kernel):
        """Test clearing cache."""
        # Cache some kernels
        cache_manager.cache_kernel("model1", "op1", sample_kernel)
        cache_manager.cache_kernel("model2", "op2", sample_kernel)
        
        assert cache_manager.count_cached_kernels() == 2
        
        # Clear cache
        cache_manager.clear_cache()
        
        # Verify cache is empty
        assert cache_manager.count_cached_kernels() == 0
        assert not any(cache_manager.kernels_dir.iterdir())


class TestOptimizationHistory:
    """Test optimization history tracking."""

    def test_record_optimization(self, cache_manager):
        """Test recording optimization results."""
        cache_manager.record_optimization(
            model_name="llama-7b",
            operation="matmul",
            baseline_time_ms=20.0,
            optimized_time_ms=10.0,
            memory_usage_mb=256.0,
            notes="Test optimization"
        )
        
        # Verify record was created
        with sqlite3.connect(cache_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM optimization_history")
            count = cursor.fetchone()[0]
            assert count == 1

    def test_record_optimization_calculates_speedup(self, cache_manager):
        """Test that speedup is calculated correctly."""
        cache_manager.record_optimization(
            model_name="llama-7b",
            operation="matmul",
            baseline_time_ms=20.0,
            optimized_time_ms=10.0,
            memory_usage_mb=256.0
        )
        
        history = cache_manager.get_optimization_history()
        assert len(history) == 1
        assert history[0]["speedup"] == 2.0

    def test_get_optimization_history_empty(self, cache_manager):
        """Test getting history when empty."""
        history = cache_manager.get_optimization_history()
        assert history == []

    def test_get_optimization_history_all(self, cache_manager):
        """Test getting all optimization history."""
        # Record multiple optimizations
        cache_manager.record_optimization("model1", "op1", 20.0, 10.0, 256.0)
        cache_manager.record_optimization("model2", "op2", 30.0, 15.0, 512.0)
        
        history = cache_manager.get_optimization_history()
        assert len(history) == 2

    def test_get_optimization_history_filtered(self, cache_manager):
        """Test getting filtered optimization history."""
        # Record multiple optimizations
        cache_manager.record_optimization("llama-7b", "op1", 20.0, 10.0, 256.0)
        cache_manager.record_optimization("llama-7b", "op2", 30.0, 15.0, 512.0)
        cache_manager.record_optimization("mistral-7b", "op1", 25.0, 12.0, 384.0)
        
        history = cache_manager.get_optimization_history("llama-7b")
        assert len(history) == 2
        assert all(h["model_name"] == "llama-7b" for h in history)

    def test_get_optimization_history_limit(self, cache_manager):
        """Test limiting optimization history results."""
        # Record many optimizations
        for i in range(10):
            cache_manager.record_optimization(f"model{i}", "op", 20.0, 10.0, 256.0)
        
        history = cache_manager.get_optimization_history(limit=5)
        assert len(history) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_kernel_with_missing_code(self, cache_manager):
        """Test caching kernel without code field."""
        kernel_data = {"operation": "matmul"}
        kernel_id = cache_manager.cache_kernel("model", "matmul", kernel_data)
        
        # Should still work but with empty code
        assert kernel_id is not None

    def test_get_cached_kernel_missing_file(self, cache_manager, sample_kernel):
        """Test retrieving kernel when file is missing."""
        # Cache a kernel
        kernel_id = cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        
        # Delete the file
        kernel_file = cache_manager.kernels_dir / f"{kernel_id}.pkl"
        kernel_file.unlink()
        
        # Should return None
        cached_kernel = cache_manager.get_cached_kernel("llama-7b", "matmul")
        assert cached_kernel is None

    def test_list_cached_kernels_missing_files(self, cache_manager, sample_kernel):
        """Test listing kernels when some files are missing."""
        # Cache kernels
        kernel_id1 = cache_manager.cache_kernel("model1", "op1", sample_kernel)
        cache_manager.cache_kernel("model2", "op2", sample_kernel)
        
        # Delete one file
        kernel_file = cache_manager.kernels_dir / f"{kernel_id1}.pkl"
        kernel_file.unlink()
        
        # Should still list kernels but with 0 size for missing file
        kernels = cache_manager.list_cached_kernels()
        assert len(kernels) == 2
        missing_kernel = next(k for k in kernels if k["model"] == "model1")
        assert missing_kernel["size_kb"] == 0

    def test_record_optimization_zero_optimized_time(self, cache_manager):
        """Test recording optimization with zero optimized time."""
        cache_manager.record_optimization(
            model_name="model",
            operation="op",
            baseline_time_ms=20.0,
            optimized_time_ms=0.0,
            memory_usage_mb=256.0
        )
        
        history = cache_manager.get_optimization_history()
        assert history[0]["speedup"] == 0

    def test_pickle_corruption_handling(self, cache_manager, sample_kernel):
        """Test handling of corrupted pickle files."""
        # Cache a kernel
        kernel_id = cache_manager.cache_kernel("llama-7b", "matmul", sample_kernel)
        
        # Corrupt the pickle file
        kernel_file = cache_manager.kernels_dir / f"{kernel_id}.pkl"
        with open(kernel_file, 'wb') as f:
            f.write(b"corrupted data")
        
        # Should handle gracefully
        with patch('builtins.open', side_effect=pickle.UnpicklingError()):
            cached_kernel = cache_manager.get_cached_kernel("llama-7b", "matmul")
            # Implementation might return None or raise, depending on error handling
            # This test ensures no unhandled exception

    def test_concurrent_database_access(self, cache_manager, sample_kernel):
        """Test concurrent database access."""
        import threading
        
        def cache_kernel(model_num):
            cache_manager.cache_kernel(f"model{model_num}", "op", sample_kernel)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=cache_kernel, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All kernels should be cached
        assert cache_manager.count_cached_kernels() == 5