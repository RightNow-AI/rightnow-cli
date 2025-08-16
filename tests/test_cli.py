"""Test suite for the CLI module."""
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from typer.testing import CliRunner
import sys
import json

from ultrathink_cli.cli import app, main


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_cache_manager():
    """Create a mock CacheManager."""
    with patch('ultrathink_cli.cli.CacheManager') as mock_cache:
        cache_instance = MagicMock()
        mock_cache.return_value = cache_instance
        yield cache_instance


@pytest.fixture
def mock_openrouter_client():
    """Create a mock OpenRouterClient."""
    with patch('ultrathink_cli.cli.OpenRouterClient') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_model_loader():
    """Create a mock ModelLoader."""
    with patch('ultrathink_cli.cli.ModelLoader') as mock_loader:
        loader_instance = MagicMock()
        mock_loader.return_value = loader_instance
        loader_instance.load_model.return_value = {
            'name': 'test-model',
            'hidden_size': 4096,
            'num_layers': 32,
            'dtype': 'float32',
            'batch_size': 1,
            'max_seq_len': 512,
            'num_attention_heads': 32
        }
        yield loader_instance


@pytest.fixture
def mock_cuda_compiler():
    """Create a mock CUDACompiler."""
    with patch('ultrathink_cli.cli.CUDACompiler') as mock_compiler:
        compiler_instance = MagicMock()
        mock_compiler.return_value = compiler_instance
        compiler_instance.compile_kernel.return_value = {
            'operation': 'matmul',
            'ptx_code': 'mock_ptx_code',
            'full_code': 'mock_cuda_code'
        }
        yield compiler_instance


@pytest.fixture
def mock_correctness_checker():
    """Create a mock CorrectnessChecker."""
    with patch('ultrathink_cli.cli.CorrectnessChecker') as mock_checker:
        checker_instance = MagicMock()
        mock_checker.return_value = checker_instance
        checker_instance.check_kernel.return_value = True
        yield checker_instance


@pytest.fixture
def mock_benchmarker():
    """Create a mock Benchmarker."""
    with patch('ultrathink_cli.cli.Benchmarker') as mock_bench:
        bench_instance = MagicMock()
        mock_bench.return_value = bench_instance
        bench_instance.benchmark_kernel.return_value = {
            'avg_time_ms': 10.5,
            'baseline_time_ms': 20.0,
            'speedup': 1.9,
            'throughput_gbps': 150.0
        }
        yield bench_instance


@pytest.fixture
def mock_cuda_profiler():
    """Create a mock CUDAProfiler."""
    with patch('ultrathink_cli.cli.CUDAProfiler') as mock_profiler:
        profiler_instance = MagicMock()
        mock_profiler.return_value = profiler_instance
        profiler_instance.profile_kernel.return_value = {
            'memory_usage_mb': 256.0,
            'occupancy': 0.85,
            'register_usage': 48
        }
        yield profiler_instance


class TestOptimizeCommand:
    """Test the optimize command."""

    def test_optimize_success(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test successful optimization."""
        # Setup mocks
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cache_manager.get_cached_kernel.return_value = None
        
        mock_kernel = {
            'code': '__global__ void test_kernel() {}',
            'operation': 'matmul',
            'constraints': {'max_registers': 255}
        }
        mock_openrouter_client.generate_kernels.return_value = [mock_kernel]
        
        # Run command
        result = runner.invoke(app, ["optimize", "mistral-7b"])
        
        # Assertions
        assert result.exit_code == 0
        assert "Optimization Results" in result.stdout
        mock_model_loader.load_model.assert_called_once_with("mistral-7b")
        mock_openrouter_client.generate_kernels.assert_called()
        mock_cuda_compiler.compile_kernel.assert_called()
        mock_correctness_checker.check_kernel.assert_called()
        mock_benchmarker.benchmark_kernel.assert_called()
        mock_cuda_profiler.profile_kernel.assert_called()

    def test_optimize_with_missing_api_key(self, runner, mock_cache_manager):
        """Test optimization when API key is missing."""
        mock_cache_manager.has_api_key.return_value = False
        
        # Simulate user input for API key
        result = runner.invoke(app, ["optimize", "mistral-7b"], input="test-api-key\n")
        
        assert result.exit_code == 0
        assert "OpenRouter API key not found" in result.stdout
        assert "API key saved successfully" in result.stdout
        mock_cache_manager.save_api_key.assert_called_once_with("test-api-key")

    def test_optimize_with_specific_operations(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test optimization with specific operations."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        result = runner.invoke(app, [
            "optimize",
            "bert-base",
            "--ops", "matmul",
            "--ops", "layernorm"
        ])
        
        assert result.exit_code == 0
        # Verify that only specified operations were processed
        calls = mock_openrouter_client.generate_kernels.call_args_list
        operations = [call[1]['operation'] for call in calls]
        assert set(operations) == {'matmul', 'layernorm'}

    def test_optimize_with_cached_kernels(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test optimization with cached kernels."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        # Return cached kernel for matmul
        cached_kernel = {
            'code': '__global__ void cached_kernel() {}',
            'operation': 'matmul'
        }
        mock_cache_manager.get_cached_kernel.side_effect = lambda model, op: (
            cached_kernel if op == 'matmul' else None
        )
        
        result = runner.invoke(app, ["optimize", "gpt2"])
        
        assert result.exit_code == 0
        assert "Using cached kernel for matmul" in result.stdout

    def test_optimize_force_regenerate(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test optimization with force regenerate flag."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_cache_manager.get_cached_kernel.return_value = {'code': 'cached'}
        
        result = runner.invoke(app, ["optimize", "llama-7b", "--force"])
        
        assert result.exit_code == 0
        # Should not use cached kernels
        assert "Using cached kernel" not in result.stdout
        # Should generate new kernels
        assert mock_openrouter_client.generate_kernels.called

    def test_optimize_compilation_failure(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test handling of compilation failures."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        # Make compilation fail
        mock_cuda_compiler.compile_kernel.side_effect = Exception("Compilation error")
        mock_kernel = {'code': 'invalid', 'operation': 'matmul'}
        mock_openrouter_client.generate_kernels.return_value = [mock_kernel]
        
        result = runner.invoke(app, ["optimize", "test-model"])
        
        assert result.exit_code == 0
        assert "Failed to compile kernel" in result.stdout

    def test_optimize_correctness_failure(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test handling of correctness check failures."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        # Make correctness check fail
        mock_correctness_checker.check_kernel.return_value = False
        
        result = runner.invoke(app, ["optimize", "test-model"])
        
        assert result.exit_code == 0
        assert "Kernel failed correctness test" in result.stdout

    def test_optimize_keyboard_interrupt(
        self,
        runner,
        mock_cache_manager,
        mock_model_loader
    ):
        """Test handling of keyboard interrupt."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        # Simulate keyboard interrupt
        mock_model_loader.load_model.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(app, ["optimize", "test-model"])
        
        assert result.exit_code == 0
        assert "Optimization cancelled by user" in result.stdout

    def test_optimize_general_error(
        self,
        runner,
        mock_cache_manager,
        mock_model_loader
    ):
        """Test handling of general errors."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        # Simulate general error
        mock_model_loader.load_model.side_effect = Exception("Test error")
        
        result = runner.invoke(app, ["optimize", "test-model"])
        
        assert result.exit_code == 1
        assert "Error: Test error" in result.stdout


class TestConfigCommand:
    """Test the config command."""

    def test_config_show(self, runner, mock_cache_manager):
        """Test showing configuration."""
        mock_cache_manager.get_config.return_value = {'openrouter_api_key': 'xxx'}
        mock_cache_manager.config_dir = Path("/home/user/.ultrathink-cli")
        mock_cache_manager.cache_dir = Path("/home/user/.ultrathink-cli/cache")
        mock_cache_manager.count_cached_kernels.return_value = 5
        
        result = runner.invoke(app, ["config", "--show"])
        
        assert result.exit_code == 0
        assert "Current Configuration" in result.stdout
        assert "Config directory:" in result.stdout
        assert "Cache directory:" in result.stdout
        assert "API key configured: Yes" in result.stdout
        assert "Cached kernels: 5" in result.stdout

    def test_config_reset_api_key(self, runner, mock_cache_manager):
        """Test resetting API key."""
        result = runner.invoke(app, ["config", "--reset-api-key"], input="new-api-key\n")
        
        assert result.exit_code == 0
        assert "API key updated successfully" in result.stdout
        mock_cache_manager.save_api_key.assert_called_once_with("new-api-key")

    def test_config_clear_cache_confirmed(self, runner, mock_cache_manager):
        """Test clearing cache with confirmation."""
        result = runner.invoke(app, ["config", "--clear-cache"], input="y\n")
        
        assert result.exit_code == 0
        assert "Cache cleared successfully" in result.stdout
        mock_cache_manager.clear_cache.assert_called_once()

    def test_config_clear_cache_cancelled(self, runner, mock_cache_manager):
        """Test clearing cache cancelled."""
        result = runner.invoke(app, ["config", "--clear-cache"], input="n\n")
        
        assert result.exit_code == 0
        assert "Cache cleared successfully" not in result.stdout
        mock_cache_manager.clear_cache.assert_not_called()

    def test_config_multiple_options(self, runner, mock_cache_manager):
        """Test multiple config options."""
        mock_cache_manager.get_config.return_value = {}
        mock_cache_manager.count_cached_kernels.return_value = 0
        
        result = runner.invoke(app, [
            "config",
            "--show",
            "--reset-api-key"
        ], input="new-key\n")
        
        assert result.exit_code == 0
        assert "Current Configuration" in result.stdout
        assert "API key updated successfully" in result.stdout


class TestListKernelsCommand:
    """Test the list-kernels command."""

    def test_list_kernels_no_results(self, runner, mock_cache_manager):
        """Test listing kernels with no results."""
        mock_cache_manager.list_cached_kernels.return_value = []
        
        result = runner.invoke(app, ["list-kernels"])
        
        assert result.exit_code == 0
        assert "No cached kernels found" in result.stdout

    def test_list_kernels_with_results(self, runner, mock_cache_manager):
        """Test listing kernels with results."""
        mock_cache_manager.list_cached_kernels.return_value = [
            {
                'model': 'llama-7b',
                'operation': 'matmul',
                'created': '2024-01-01T12:00:00',
                'size_kb': 12.5
            },
            {
                'model': 'mistral-7b',
                'operation': 'attention',
                'created': '2024-01-02T14:00:00',
                'size_kb': 25.3
            }
        ]
        
        result = runner.invoke(app, ["list-kernels"])
        
        assert result.exit_code == 0
        assert "llama-7b" in result.stdout
        assert "matmul" in result.stdout
        assert "mistral-7b" in result.stdout
        assert "attention" in result.stdout

    def test_list_kernels_filtered_by_model(self, runner, mock_cache_manager):
        """Test listing kernels filtered by model."""
        mock_cache_manager.list_cached_kernels.return_value = [
            {
                'model': 'llama-7b',
                'operation': 'matmul',
                'created': '2024-01-01T12:00:00',
                'size_kb': 12.5
            }
        ]
        
        result = runner.invoke(app, ["list-kernels", "--model", "llama-7b"])
        
        assert result.exit_code == 0
        assert "llama-7b" in result.stdout
        mock_cache_manager.list_cached_kernels.assert_called_once_with("llama-7b")


class TestVersionCommand:
    """Test the version command."""

    def test_version_success(self, runner):
        """Test version command success."""
        with patch('ultrathink_cli.cli.version') as mock_version:
            mock_version.return_value = "1.0.0"
            result = runner.invoke(app, ["version"])
            
            assert result.exit_code == 0
            assert "ultrathink-cli version" in result.stdout

    def test_version_development(self, runner):
        """Test version command in development mode."""
        with patch('ultrathink_cli.cli.version') as mock_version:
            mock_version.side_effect = Exception("Not installed")
            result = runner.invoke(app, ["version"])
            
            assert result.exit_code == 0
            assert "ultrathink-cli version 0.1.0 (development)" in result.stdout


class TestMainFunction:
    """Test the main function."""

    def test_main_function(self):
        """Test main function calls app."""
        with patch('ultrathink_cli.cli.app') as mock_app:
            main()
            mock_app.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_optimize_custom_iterations(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test optimization with custom iterations."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        result = runner.invoke(app, [
            "optimize",
            "test-model",
            "--iterations", "50"
        ])
        
        assert result.exit_code == 0
        mock_benchmarker.assert_called_with(iterations=50)

    def test_optimize_custom_tolerance(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test optimization with custom tolerance."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        
        result = runner.invoke(app, [
            "optimize",
            "test-model",
            "--tolerance", "1e-6"
        ])
        
        assert result.exit_code == 0
        mock_correctness_checker.assert_called_with(tolerance=1e-6)

    def test_optimize_no_valid_kernels(
        self,
        runner,
        mock_cache_manager,
        mock_openrouter_client,
        mock_model_loader,
        mock_cuda_compiler,
        mock_correctness_checker,
        mock_benchmarker,
        mock_cuda_profiler
    ):
        """Test optimization when no kernels pass correctness check."""
        mock_cache_manager.has_api_key.return_value = True
        mock_cache_manager.get_api_key.return_value = "test-api-key"
        mock_correctness_checker.check_kernel.return_value = False
        
        result = runner.invoke(app, ["optimize", "test-model"])
        
        assert result.exit_code == 0
        # Should complete but with no results in the table
        assert "Optimization Results" in result.stdout