"""Test suite for the Benchmarking module."""
import pytest
from unittest.mock import MagicMock, patch, Mock, PropertyMock
import numpy as np
import time
import statistics

from ultrathink_cli.bench import Benchmarker


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability."""
    with patch('ultrathink_cli.bench.torch.cuda.is_available', return_value=True):
        yield


@pytest.fixture
def mock_cuda_not_available():
    """Mock CUDA not available."""
    with patch('ultrathink_cli.bench.torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def mock_torch():
    """Mock torch module."""
    with patch('ultrathink_cli.bench.torch') as mock_torch:
        # Setup torch mock
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        mock_torch.float32 = "float32"
        
        # Mock tensor operations
        mock_tensor = MagicMock()
        mock_tensor.shape = (100, 100)
        mock_tensor.nbytes = 40000
        mock_tensor.cpu.return_value.numpy.return_value = np.ones((100, 100))
        
        mock_torch.randn.return_value = mock_tensor
        mock_torch.zeros.return_value = mock_tensor
        mock_torch.ones.return_value = mock_tensor
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.matmul.return_value = mock_tensor
        mock_torch.nn.functional.layer_norm.return_value = mock_tensor
        mock_torch.nn.functional.gelu.return_value = mock_tensor
        mock_torch.nn.functional.softmax.return_value = mock_tensor
        
        yield mock_torch


@pytest.fixture
def mock_cuda():
    """Mock pycuda module."""
    with patch('ultrathink_cli.bench.cuda') as mock_cuda:
        # Mock CUDA device and context
        mock_device = MagicMock()
        mock_context = MagicMock()
        mock_device.make_context.return_value = mock_context
        mock_cuda.Device.return_value = mock_device
        mock_cuda.init.return_value = None
        
        # Mock CUDA memory operations
        mock_cuda.mem_alloc.return_value = MagicMock()
        mock_cuda.memcpy_htod.return_value = None
        mock_cuda.Context.synchronize.return_value = None
        
        # Mock CUDA events
        mock_event = MagicMock()
        mock_event.time_till.return_value = 10.5  # milliseconds
        mock_cuda.Event.return_value = mock_event
        
        yield mock_cuda


@pytest.fixture
def mock_source_module():
    """Mock SourceModule for CUDA compilation."""
    with patch('ultrathink_cli.bench.SourceModule') as mock_sm:
        mock_module = MagicMock()
        mock_kernel_func = MagicMock()
        mock_module.get_function.return_value = mock_kernel_func
        mock_sm.return_value = mock_module
        yield mock_sm, mock_module, mock_kernel_func


@pytest.fixture
def benchmarker(mock_cuda_available, mock_torch, mock_cuda):
    """Create a Benchmarker instance with mocked dependencies."""
    return Benchmarker(iterations=10, warmup_iterations=2)


@pytest.fixture
def compiled_kernel():
    """Create a sample compiled kernel."""
    return {
        "operation": "matmul",
        "ptx_code": "mock_ptx_code",
        "full_code": "__global__ void matmul_kernel() {}"
    }


@pytest.fixture
def model_info():
    """Create sample model information."""
    return {
        'name': 'test-model',
        'hidden_size': 4096,
        'num_layers': 32,
        'dtype': 'float32',
        'batch_size': 1,
        'max_seq_len': 512,
        'num_attention_heads': 32
    }


class TestBenchmarkerInitialization:
    """Test Benchmarker initialization."""

    def test_initialization_with_cuda(self, mock_cuda_available, mock_torch, mock_cuda):
        """Test initialization when CUDA is available."""
        benchmarker = Benchmarker(iterations=50, warmup_iterations=5)
        
        assert benchmarker.iterations == 50
        assert benchmarker.warmup_iterations == 5
        assert benchmarker.device == "cuda"
        mock_cuda.init.assert_called_once()

    def test_initialization_without_cuda(self, mock_cuda_not_available):
        """Test initialization when CUDA is not available."""
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            Benchmarker()

    def test_context_cleanup(self, mock_cuda_available, mock_torch, mock_cuda):
        """Test CUDA context cleanup on deletion."""
        benchmarker = Benchmarker()
        mock_context = benchmarker.cuda_context
        
        # Simulate deletion
        del benchmarker
        
        # Verify context was popped
        mock_context.pop.assert_called_once()


class TestBenchmarkKernel:
    """Test the main benchmark_kernel method."""

    def test_benchmark_kernel_matmul(self, benchmarker, compiled_kernel, model_info, mock_source_module):
        """Test benchmarking a matmul kernel."""
        with patch('ultrathink_cli.bench.console'):
            result = benchmarker.benchmark_kernel(compiled_kernel, model_info)
        
        assert "avg_time_ms" in result
        assert "baseline_time_ms" in result
        assert "speedup" in result
        assert "throughput_gbps" in result
        assert result["speedup"] > 0

    def test_benchmark_kernel_unknown_operation(self, benchmarker, model_info):
        """Test benchmarking an unknown operation."""
        unknown_kernel = {"operation": "unknown_op"}
        
        with patch('ultrathink_cli.bench.console'):
            result = benchmarker.benchmark_kernel(unknown_kernel, model_info)
        
        assert result["avg_time_ms"] == float('inf')
        assert result["baseline_time_ms"] == float('inf')

    def test_benchmark_kernel_error_handling(self, benchmarker, compiled_kernel, model_info):
        """Test error handling during benchmarking."""
        with patch.object(benchmarker, '_benchmark_matmul', side_effect=Exception("Test error")):
            with patch('ultrathink_cli.bench.console'):
                result = benchmarker.benchmark_kernel(compiled_kernel, model_info)
        
        assert result["avg_time_ms"] == float('inf')
        assert result["baseline_time_ms"] == float('inf')


class TestOperationBenchmarks:
    """Test individual operation benchmarks."""

    def test_benchmark_matmul(self, benchmarker, compiled_kernel, model_info, mock_torch):
        """Test matrix multiplication benchmarking."""
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [20.0, 20.5, 19.5]
                mock_cuda.return_value = [10.0, 10.5, 9.5]
                
                result = benchmarker._benchmark_matmul(compiled_kernel, model_info)
        
        assert "baseline_time_ms" in result
        assert "kernel_time_ms" in result
        assert "throughput_gbps" in result
        assert result["speedup"] == statistics.median([20.0, 20.5, 19.5]) / statistics.median([10.0, 10.5, 9.5])

    def test_benchmark_layernorm(self, benchmarker, compiled_kernel, model_info, mock_torch):
        """Test layer normalization benchmarking."""
        compiled_kernel["operation"] = "layernorm"
        
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [15.0, 15.5, 14.5]
                mock_cuda.return_value = [8.0, 8.5, 7.5]
                
                result = benchmarker._benchmark_layernorm(compiled_kernel, model_info)
        
        assert "baseline_time_ms" in result
        assert "kernel_time_ms" in result
        assert "throughput_gbps" in result

    def test_benchmark_gelu(self, benchmarker, compiled_kernel, model_info, mock_torch):
        """Test GELU activation benchmarking."""
        compiled_kernel["operation"] = "gelu"
        
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [12.0, 12.5, 11.5]
                mock_cuda.return_value = [6.0, 6.5, 5.5]
                
                result = benchmarker._benchmark_gelu(compiled_kernel, model_info)
        
        assert "baseline_time_ms" in result
        assert "kernel_time_ms" in result
        assert "throughput_gbps" in result

    def test_benchmark_softmax(self, benchmarker, compiled_kernel, model_info, mock_torch):
        """Test softmax benchmarking."""
        compiled_kernel["operation"] = "softmax"
        
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [18.0, 18.5, 17.5]
                mock_cuda.return_value = [9.0, 9.5, 8.5]
                
                result = benchmarker._benchmark_softmax(compiled_kernel, model_info)
        
        assert "baseline_time_ms" in result
        assert "kernel_time_ms" in result
        assert "throughput_gbps" in result

    def test_benchmark_attention(self, benchmarker, compiled_kernel, model_info, mock_torch):
        """Test attention benchmarking."""
        compiled_kernel["operation"] = "attention"
        
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [25.0, 25.5, 24.5]
                mock_cuda.return_value = [12.0, 12.5, 11.5]
                
                result = benchmarker._benchmark_attention(compiled_kernel, model_info)
        
        assert "baseline_time_ms" in result
        assert "kernel_time_ms" in result
        assert "tflops" in result


class TestPyTorchBenchmarking:
    """Test PyTorch operation benchmarking."""

    def test_benchmark_pytorch_op(self, benchmarker, mock_torch):
        """Test benchmarking a PyTorch operation."""
        op_count = 0
        def mock_op():
            nonlocal op_count
            op_count += 1
            return mock_torch.zeros(100, 100)
        
        with patch('ultrathink_cli.bench.time.perf_counter') as mock_time:
            # Mock time to simulate execution
            mock_time.side_effect = [0.0, 0.01, 0.01, 0.02, 0.02, 0.03]  # warmup + iterations
            
            with patch('ultrathink_cli.bench.console'):
                times = benchmarker._benchmark_pytorch_op(
                    mock_op,
                    warmup=2,
                    iterations=2
                )
        
        assert len(times) == 2
        assert all(t > 0 for t in times)
        assert op_count == 4  # 2 warmup + 2 iterations


class TestCUDAKernelBenchmarking:
    """Test CUDA kernel benchmarking."""

    def test_benchmark_cuda_kernel_success(self, benchmarker, compiled_kernel, mock_source_module, mock_cuda, mock_torch):
        """Test successful CUDA kernel benchmarking."""
        _, mock_module, mock_kernel_func = mock_source_module
        
        inputs = [mock_torch.randn(100, 100)]
        output_shape = (100, 100)
        
        with patch('ultrathink_cli.bench.console'):
            times = benchmarker._benchmark_cuda_kernel(
                compiled_kernel,
                inputs,
                output_shape,
                "matmul_kernel",
                (16, 16, 1),
                (10, 10, 1),
                warmup=2,
                iterations=3
            )
        
        assert len(times) == 3
        assert all(t == 10.5 for t in times)  # Mocked event time
        mock_module.get_function.assert_called_with("matmul_kernel")

    def test_benchmark_cuda_kernel_no_code(self, benchmarker):
        """Test benchmarking kernel without code."""
        empty_kernel = {}
        
        with pytest.raises(ValueError, match="No compiled kernel code available"):
            benchmarker._benchmark_cuda_kernel(
                empty_kernel,
                [],
                (100, 100),
                "kernel",
                (16, 16, 1),
                (10, 10, 1)
            )

    def test_benchmark_cuda_kernel_compilation_error(self, benchmarker, compiled_kernel, mock_source_module):
        """Test handling compilation errors."""
        mock_sm, _, _ = mock_source_module
        mock_sm.side_effect = Exception("Compilation failed")
        
        with patch('ultrathink_cli.bench.console'):
            times = benchmarker._benchmark_cuda_kernel(
                compiled_kernel,
                [],
                (100, 100),
                "kernel",
                (16, 16, 1),
                (10, 10, 1),
                iterations=3
            )
        
        assert all(t == float('inf') for t in times)

    def test_benchmark_cuda_kernel_with_ptx(self, benchmarker, mock_cuda):
        """Test benchmarking with PTX code."""
        kernel_with_ptx = {"ptx_code": "mock_ptx", "operation": "test"}
        
        mock_module = MagicMock()
        mock_kernel_func = MagicMock()
        mock_module.get_function.return_value = mock_kernel_func
        
        with patch.object(mock_cuda, 'module_from_buffer', return_value=mock_module):
            with patch('ultrathink_cli.bench.console'):
                times = benchmarker._benchmark_cuda_kernel(
                    kernel_with_ptx,
                    [],
                    (100, 100),
                    "kernel",
                    (16, 16, 1),
                    (10, 10, 1),
                    iterations=2
                )
        
        assert len(times) == 2
        mock_cuda.module_from_buffer.assert_called_once()


class TestBenchmarkReport:
    """Test benchmark report generation."""

    def test_generate_benchmark_report_empty(self, benchmarker):
        """Test generating report with no results."""
        report = benchmarker.generate_benchmark_report([])
        
        assert report["summary"]["total_kernels"] == 0
        assert report["detailed_results"] == []

    def test_generate_benchmark_report_with_results(self, benchmarker):
        """Test generating report with results."""
        results = [
            {"speedup": 2.0, "operation": "matmul"},
            {"speedup": 1.5, "operation": "layernorm"},
            {"speedup": 3.0, "operation": "gelu"}
        ]
        
        report = benchmarker.generate_benchmark_report(results)
        
        assert report["summary"]["total_kernels"] == 3
        assert report["summary"]["average_speedup"] == 2.166666666666667
        assert report["summary"]["best_speedup"] == 3.0
        assert report["summary"]["worst_speedup"] == 1.5
        assert report["detailed_results"] == results


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_benchmark_with_single_time_measurement(self, benchmarker):
        """Test statistics calculations with single measurement."""
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [20.0]  # Single measurement
                mock_cuda.return_value = [10.0]
                
                compiled_kernel = {"operation": "matmul"}
                model_info = {"batch_size": 1, "hidden_size": 100}
                
                result = benchmarker._benchmark_matmul(compiled_kernel, model_info)
        
        # Should handle single measurement without error
        assert result["baseline_std_ms"] == 0
        assert result["kernel_std_ms"] == 0

    def test_benchmark_pytorch_op_with_exception(self, benchmarker, mock_torch):
        """Test PyTorch benchmarking when operation throws exception."""
        def failing_op():
            raise RuntimeError("Operation failed")
        
        # Should propagate exception
        with pytest.raises(RuntimeError):
            benchmarker._benchmark_pytorch_op(failing_op, warmup=1, iterations=1)

    def test_cuda_memory_allocation_handling(self, benchmarker, compiled_kernel, mock_cuda, mock_source_module):
        """Test handling of CUDA memory allocation."""
        _, mock_module, _ = mock_source_module
        
        # Test with tensor inputs
        tensor_input = MagicMock()
        tensor_input.nbytes = 1000
        tensor_input.cpu.return_value.numpy.return_value = np.ones((10, 10))
        
        inputs = [tensor_input, 3.14]  # Mixed tensor and scalar
        
        with patch('ultrathink_cli.bench.console'):
            times = benchmarker._benchmark_cuda_kernel(
                compiled_kernel,
                inputs,
                (10, 10),
                "kernel",
                (16, 16, 1),
                (1, 1, 1),
                iterations=1
            )
        
        # Should allocate memory only for tensor
        mock_cuda.mem_alloc.assert_called_once_with(1000)

    def test_benchmark_attention_with_small_sequence(self, benchmarker, compiled_kernel, model_info, mock_torch):
        """Test attention benchmarking with sequence length limit."""
        model_info["max_seq_len"] = 2048  # Large sequence length
        compiled_kernel["operation"] = "attention"
        
        with patch.object(benchmarker, '_benchmark_pytorch_op') as mock_pytorch:
            with patch.object(benchmarker, '_benchmark_cuda_kernel') as mock_cuda:
                mock_pytorch.return_value = [25.0]
                mock_cuda.return_value = [12.0]
                
                result = benchmarker._benchmark_attention(compiled_kernel, model_info)
        
        # Should limit sequence length to 128
        assert "B1_H32_S128_D" in result["problem_size"]

    def test_concurrent_benchmarking(self, mock_cuda_available, mock_torch, mock_cuda):
        """Test that benchmarker handles context properly for concurrent use."""
        benchmarker1 = Benchmarker()
        benchmarker2 = Benchmarker()
        
        # Each should have its own context
        assert benchmarker1.cuda_context != benchmarker2.cuda_context