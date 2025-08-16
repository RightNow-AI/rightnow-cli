"""Test suite for the OpenRouter module."""
import pytest
from unittest.mock import MagicMock, patch, Mock
import requests
import json
import time

from ultrathink_cli.openrouter import OpenRouterClient, KernelCandidate


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "test-api-key-123"


@pytest.fixture
def openrouter_client(api_key):
    """Create an OpenRouterClient instance."""
    return OpenRouterClient(api_key)


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for API calls."""
    with patch('ultrathink_cli.openrouter.requests.post') as mock_post:
        yield mock_post


@pytest.fixture
def successful_api_response():
    """Create a successful API response."""
    return {
        "choices": [{
            "message": {
                "content": """Here's an optimized CUDA kernel for matrix multiplication:

```cuda
#include <cuda_runtime.h>

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

This kernel uses coalesced memory access patterns for optimal performance.
"""
            }
        }]
    }


@pytest.fixture
def model_info():
    """Create test model information."""
    return {
        'name': 'test-model',
        'hidden_size': 4096,
        'num_layers': 32,
        'dtype': 'float32',
        'batch_size': 1,
        'max_seq_len': 512,
        'num_attention_heads': 32
    }


@pytest.fixture
def constraints():
    """Create test constraints."""
    return {
        'max_registers': 255,
        'shared_memory_kb': 48,
        'block_size': 256
    }


class TestOpenRouterClient:
    """Test the OpenRouterClient class."""

    def test_initialization(self, api_key):
        """Test client initialization."""
        client = OpenRouterClient(api_key)
        
        assert client.api_key == api_key
        assert client.headers["Authorization"] == f"Bearer {api_key}"
        assert client.headers["Content-Type"] == "application/json"
        assert "HTTP-Referer" in client.headers
        assert "X-Title" in client.headers

    def test_make_request_success(self, openrouter_client, mock_requests_post, successful_api_response):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = successful_api_response
        mock_requests_post.return_value = mock_response
        
        result = openrouter_client._make_request(
            "Generate a kernel",
            "You are an expert",
            "openai/gpt-4"
        )
        
        assert "matmul_kernel" in result
        mock_requests_post.assert_called_once()
        
        # Verify request payload
        call_args = mock_requests_post.call_args
        payload = call_args.kwargs['json']
        assert payload['model'] == "openai/gpt-4"
        assert len(payload['messages']) == 2
        assert payload['messages'][0]['role'] == 'system'
        assert payload['messages'][1]['role'] == 'user'

    def test_make_request_401_error(self, openrouter_client, mock_requests_post):
        """Test handling of 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_requests_post.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid API key"):
            openrouter_client._make_request("Test", "System", "model")

    def test_make_request_other_http_error(self, openrouter_client, mock_requests_post):
        """Test handling of other HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        mock_requests_post.return_value = mock_response
        
        # Should be caught by backoff decorator
        with pytest.raises(requests.HTTPError):
            openrouter_client._make_request("Test", "System", "model")

    def test_make_request_timeout(self, openrouter_client, mock_requests_post):
        """Test handling of request timeout."""
        mock_requests_post.side_effect = requests.Timeout("Request timed out")
        
        # Should be caught by backoff decorator
        with pytest.raises(requests.Timeout):
            openrouter_client._make_request("Test", "System", "model")

    def test_generate_kernels_success(self, openrouter_client, mock_requests_post, successful_api_response, model_info, constraints):
        """Test successful kernel generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = successful_api_response
        mock_requests_post.return_value = mock_response
        
        with patch('ultrathink_cli.openrouter.console'):  # Mock console output
            candidates = openrouter_client.generate_kernels(
                operation="matmul",
                model_info=model_info,
                constraints=constraints,
                num_candidates=2
            )
        
        assert len(candidates) == 2
        assert all(isinstance(c, KernelCandidate) for c in candidates)
        assert candidates[0].operation == "matmul"
        assert candidates[0].constraints == constraints
        assert "__global__ void matmul_kernel" in candidates[0].code

    def test_generate_kernels_extraction_failure(self, openrouter_client, mock_requests_post, model_info, constraints):
        """Test handling of failed code extraction."""
        # Response without valid CUDA code
        bad_response = {
            "choices": [{
                "message": {
                    "content": "Here's some text without any CUDA code."
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = bad_response
        mock_requests_post.return_value = mock_response
        
        with patch('ultrathink_cli.openrouter.console'):  # Mock console output
            candidates = openrouter_client.generate_kernels(
                operation="matmul",
                model_info=model_info,
                constraints=constraints,
                num_candidates=1
            )
        
        assert len(candidates) == 0

    def test_generate_kernels_api_error(self, openrouter_client, mock_requests_post, model_info, constraints):
        """Test handling of API errors during kernel generation."""
        mock_requests_post.side_effect = Exception("API Error")
        
        with patch('ultrathink_cli.openrouter.console'):  # Mock console output
            candidates = openrouter_client.generate_kernels(
                operation="matmul",
                model_info=model_info,
                constraints=constraints,
                num_candidates=1
            )
        
        assert len(candidates) == 0

    def test_extract_cuda_code_with_markdown(self, openrouter_client):
        """Test CUDA code extraction from markdown."""
        response = """Here's the kernel:
```cuda
__global__ void test_kernel() {
    // kernel code
}
```
"""
        code = openrouter_client._extract_cuda_code(response)
        assert code is not None
        assert "__global__ void test_kernel()" in code
        assert "// kernel code" in code

    def test_extract_cuda_code_with_cpp_marker(self, openrouter_client):
        """Test CUDA code extraction with cpp marker."""
        response = """Here's the kernel:
```cpp
__global__ void test_kernel() {
    int idx = threadIdx.x;
}
```
"""
        code = openrouter_client._extract_cuda_code(response)
        assert code is not None
        assert "int idx = threadIdx.x;" in code

    def test_extract_cuda_code_without_markdown(self, openrouter_client):
        """Test CUDA code extraction without markdown."""
        response = """Here's the kernel:
#include <cuda.h>

__global__ void test_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float helper() {
    return 1.0f;
}
"""
        code = openrouter_client._extract_cuda_code(response)
        assert code is not None
        assert "#include <cuda.h>" in code
        assert "__global__ void test_kernel()" in code
        assert "__device__ float helper()" in code

    def test_extract_cuda_code_no_valid_code(self, openrouter_client):
        """Test extraction when no valid CUDA code is present."""
        response = "This response contains no CUDA code at all."
        code = openrouter_client._extract_cuda_code(response)
        assert code is None

    def test_validate_api_key_valid(self, openrouter_client, mock_requests_post):
        """Test API key validation with valid key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello"}}]}
        mock_requests_post.return_value = mock_response
        
        assert openrouter_client.validate_api_key() is True

    def test_validate_api_key_invalid(self, openrouter_client, mock_requests_post):
        """Test API key validation with invalid key."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_requests_post.return_value = mock_response
        
        with patch.object(openrouter_client, '_make_request') as mock_request:
            mock_request.side_effect = ValueError("Invalid API key")
            assert openrouter_client.validate_api_key() is False

    def test_validate_api_key_network_error(self, openrouter_client, mock_requests_post):
        """Test API key validation with network error."""
        mock_requests_post.side_effect = requests.ConnectionError()
        
        assert openrouter_client.validate_api_key() is False


class TestKernelPrompts:
    """Test kernel-specific prompt generation."""

    def test_matmul_prompt(self, openrouter_client, model_info):
        """Test matrix multiplication prompt generation."""
        prompt = openrouter_client._matmul_prompt(model_info)
        
        assert "matrix multiplication" in prompt
        assert "GEMM" in prompt
        assert "4096" in prompt  # hidden_size
        assert "Shared memory tiling" in prompt
        assert "Tensor cores" in prompt

    def test_attention_prompt(self, openrouter_client, model_info):
        """Test attention prompt generation."""
        prompt = openrouter_client._attention_prompt(model_info)
        
        assert "scaled dot-product attention" in prompt
        assert "32" in prompt  # num_attention_heads
        assert "Flash Attention" in prompt
        assert "softmax" in prompt

    def test_layernorm_prompt(self, openrouter_client, model_info):
        """Test layer normalization prompt generation."""
        prompt = openrouter_client._layernorm_prompt(model_info)
        
        assert "layer normalization" in prompt
        assert "4096" in prompt  # hidden_size
        assert "Welford's algorithm" in prompt
        assert "gamma" in prompt
        assert "beta" in prompt

    def test_gelu_prompt(self, openrouter_client, model_info):
        """Test GELU activation prompt generation."""
        prompt = openrouter_client._gelu_prompt(model_info)
        
        assert "GELU" in prompt
        assert "Gaussian Error Linear Unit" in prompt
        assert "tanh-based" in prompt
        assert "polynomial" in prompt

    def test_softmax_prompt(self, openrouter_client, model_info):
        """Test softmax prompt generation."""
        prompt = openrouter_client._softmax_prompt(model_info)
        
        assert "softmax" in prompt
        assert "numerical stability" in prompt
        assert "subtract max" in prompt
        assert "Online softmax" in prompt


class TestPromptCreation:
    """Test prompt creation methods."""

    def test_create_system_prompt(self, openrouter_client):
        """Test system prompt creation."""
        prompt = openrouter_client._create_system_prompt()
        
        assert "expert CUDA kernel developer" in prompt
        assert "high-performance GPU computing" in prompt
        assert "numerically correct" in prompt
        assert "PyTorch tensors" in prompt
        assert "CUDA best practices" in prompt

    def test_create_kernel_generation_prompt(self, openrouter_client, model_info, constraints):
        """Test kernel generation prompt creation."""
        prompt = openrouter_client._create_kernel_generation_prompt(
            "matmul",
            model_info,
            constraints
        )
        
        assert "matrix multiplication" in prompt
        assert "test-model" in prompt
        assert "Hidden size: 4096" in prompt
        assert "max_registers: 255" in prompt
        assert "shared_memory_kb: 48" in prompt
        assert "numerically stable" in prompt

    def test_create_kernel_generation_prompt_unknown_op(self, openrouter_client, model_info, constraints):
        """Test prompt creation for unknown operation."""
        prompt = openrouter_client._create_kernel_generation_prompt(
            "custom_op",
            model_info,
            constraints
        )
        
        assert "optimized CUDA kernel for custom_op" in prompt
        assert "Model Information:" in prompt
        assert "Constraints:" in prompt


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_generate_kernels_zero_candidates(self, openrouter_client, model_info, constraints):
        """Test generating zero kernel candidates."""
        with patch('ultrathink_cli.openrouter.console'):
            candidates = openrouter_client.generate_kernels(
                operation="test",
                model_info=model_info,
                constraints=constraints,
                num_candidates=0
            )
        
        assert len(candidates) == 0

    def test_extract_cuda_code_multiple_blocks(self, openrouter_client):
        """Test extraction with multiple code blocks."""
        response = """Here are two kernels:
```cuda
__global__ void kernel1() {
    // First kernel
}
```

And another:
```cuda
__global__ void kernel2() {
    // Second kernel
}
```
"""
        code = openrouter_client._extract_cuda_code(response)
        # Should extract the first code block
        assert code is not None
        assert "kernel1" in code
        assert "kernel2" not in code

    def test_backoff_decorator_retries(self, openrouter_client, mock_requests_post):
        """Test that backoff decorator retries on failure."""
        # First two calls fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = requests.HTTPError()
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"choices": [{"message": {"content": "Success"}}]}
        
        mock_requests_post.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success
        ]
        
        result = openrouter_client._make_request("Test", "System", "model")
        assert result == "Success"
        assert mock_requests_post.call_count == 3

    def test_kernel_candidate_dataclass(self):
        """Test KernelCandidate dataclass."""
        candidate = KernelCandidate(
            code="__global__ void test() {}",
            operation="test_op",
            constraints={"max_registers": 64},
            metadata={"version": 1}
        )
        
        assert candidate.code == "__global__ void test() {}"
        assert candidate.operation == "test_op"
        assert candidate.constraints["max_registers"] == 64
        assert candidate.metadata["version"] == 1