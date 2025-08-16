import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from typing import Dict, Any, Tuple, List, Optional
from rich.console import Console
import time

console = Console()


class CorrectnessChecker:
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot perform correctness checks.")
        
        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
    
    def __del__(self):
        """Clean up CUDA context."""
        if hasattr(self, 'cuda_context'):
            self.cuda_context.pop()
    
    def check_kernel_against_original(
        self,
        optimized_kernel: Dict[str, Any],
        original_code: str,
        test_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if optimized kernel produces same results as original."""
        console.print("[cyan]Testing kernel correctness against original...[/cyan]")
        
        try:
            # Compile original kernel
            original_compiled = {
                "code": original_code,
                "operation": "original",
                "full_code": original_code
            }
            
            # If no test data provided, create synthetic data
            if not test_data:
                test_data = self._create_default_test_data()
            
            # Run both kernels
            original_output = self._run_cuda_kernel(
                original_compiled,
                inputs=test_data['inputs'],
                output_shape=test_data['output_shape'],
                kernel_name=test_data.get('kernel_name', 'kernel'),
                block_size=test_data.get('block_size', (256, 1, 1)),
                grid_size=test_data.get('grid_size', (1024, 1, 1))
            )
            
            optimized_output = self._run_cuda_kernel(
                optimized_kernel,
                inputs=test_data['inputs'],
                output_shape=test_data['output_shape'],
                kernel_name=test_data.get('kernel_name', 'kernel'),
                block_size=test_data.get('block_size', (256, 1, 1)),
                grid_size=test_data.get('grid_size', (1024, 1, 1))
            )
            
            # Compare outputs
            error_stats = self._compute_error_stats(original_output, optimized_output)
            
            passed = error_stats['max_abs_error'] < self.tolerance
            
            if passed:
                console.print(f"[green]✓ Correctness test passed[/green]")
                console.print(f"  Max absolute error: {error_stats['max_abs_error']:.2e}")
            else:
                console.print(f"[red]✗ Correctness test failed[/red]")
                console.print(f"  Max absolute error: {error_stats['max_abs_error']:.2e} (tolerance: {self.tolerance})")
            
            return passed
            
        except Exception as e:
            console.print(f"[red]Error during correctness test: {e}[/red]")
            return False
    
    def _create_default_test_data(self) -> Dict[str, Any]:
        """Create default test data for kernel testing."""
        size = 1024
        return {
            'inputs': [torch.randn(size, size, dtype=torch.float32, device=self.device)],
            'output_shape': (size, size),
            'kernel_name': 'kernel',
            'block_size': (16, 16, 1),
            'grid_size': ((size + 15) // 16, (size + 15) // 16, 1)
        }
    
    def check_kernel(self, compiled_kernel: Dict[str, Any], model_info: Dict[str, Any]) -> bool:
        """Check if a kernel produces correct results compared to PyTorch baseline."""
        operation = compiled_kernel.get("operation", "unknown")
        
        console.print(f"[cyan]Testing correctness for {operation} kernel...[/cyan]")
        
        test_functions = {
            "matmul": self._test_matmul_kernel,
            "layernorm": self._test_layernorm_kernel,
            "gelu": self._test_gelu_kernel,
            "softmax": self._test_softmax_kernel,
            "attention": self._test_attention_kernel
        }
        
        if operation not in test_functions:
            console.print(f"[yellow]No correctness test available for {operation}[/yellow]")
            return True
        
        try:
            test_func = test_functions[operation]
            passed, error_stats = test_func(compiled_kernel, model_info)
            
            if passed:
                console.print(f"[green] Correctness test passed for {operation}[/green]")
                console.print(f"  Max absolute error: {error_stats['max_abs_error']:.2e}")
                console.print(f"  Mean absolute error: {error_stats['mean_abs_error']:.2e}")
            else:
                console.print(f"[red] Correctness test failed for {operation}[/red]")
                console.print(f"  Max absolute error: {error_stats['max_abs_error']:.2e} (tolerance: {self.tolerance})")
            
            return passed
            
        except Exception as e:
            console.print(f"[red]Error during correctness test: {e}[/red]")
            return False
    
    def _test_matmul_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, float]]:
        """Test matrix multiplication kernel correctness."""
        batch_size = model_info.get("batch_size", 1)
        hidden_size = model_info.get("hidden_size", 4096)
        
        test_cases = [
            (batch_size, hidden_size, hidden_size),
            (batch_size, hidden_size, hidden_size * 4),
            (batch_size, hidden_size * 4, hidden_size),
        ]
        
        all_errors = []
        
        for m, k, n in test_cases:
            a_torch = torch.randn(m, k, dtype=torch.float32, device=self.device)
            b_torch = torch.randn(k, n, dtype=torch.float32, device=self.device)
            c_torch = torch.zeros(m, n, dtype=torch.float32, device=self.device)
            
            expected = torch.matmul(a_torch, b_torch)
            
            actual = self._run_cuda_kernel(
                compiled_kernel,
                inputs=[a_torch, b_torch, c_torch],
                output_shape=(m, n),
                kernel_name="matmul_kernel",
                block_size=(16, 16, 1),
                grid_size=((n + 15) // 16, (m + 15) // 16, 1)
            )
            
            error_stats = self._compute_error_stats(expected, actual)
            all_errors.append(error_stats)
        
        max_error = max(e['max_abs_error'] for e in all_errors)
        mean_error = np.mean([e['mean_abs_error'] for e in all_errors])
        
        return max_error < self.tolerance, {
            'max_abs_error': max_error,
            'mean_abs_error': mean_error
        }
    
    def _test_layernorm_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, float]]:
        """Test layer normalization kernel correctness."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        hidden_size = model_info.get("hidden_size", 4096)
        
        input_torch = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=self.device)
        gamma = torch.ones(hidden_size, dtype=torch.float32, device=self.device)
        beta = torch.zeros(hidden_size, dtype=torch.float32, device=self.device)
        
        eps = 1e-5
        expected = torch.nn.functional.layer_norm(input_torch, (hidden_size,), gamma, beta, eps)
        
        actual = self._run_cuda_kernel(
            compiled_kernel,
            inputs=[input_torch, gamma, beta, torch.tensor(eps)],
            output_shape=(batch_size, seq_len, hidden_size),
            kernel_name="layernorm_kernel",
            block_size=(256, 1, 1),
            grid_size=(seq_len, batch_size, 1)
        )
        
        error_stats = self._compute_error_stats(expected, actual)
        
        return error_stats['max_abs_error'] < self.tolerance, error_stats
    
    def _test_gelu_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, float]]:
        """Test GELU activation kernel correctness."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        hidden_size = model_info.get("hidden_size", 4096)
        
        input_torch = torch.randn(
            batch_size * seq_len * hidden_size,
            dtype=torch.float32,
            device=self.device
        )
        
        expected = torch.nn.functional.gelu(input_torch)
        
        actual = self._run_cuda_kernel(
            compiled_kernel,
            inputs=[input_torch],
            output_shape=input_torch.shape,
            kernel_name="gelu_kernel",
            block_size=(256, 1, 1),
            grid_size=((input_torch.numel() + 255) // 256, 1, 1)
        )
        
        error_stats = self._compute_error_stats(expected, actual)
        
        return error_stats['max_abs_error'] < self.tolerance, error_stats
    
    def _test_softmax_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, float]]:
        """Test softmax kernel correctness."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        num_heads = model_info.get("num_attention_heads", 32)
        
        input_torch = torch.randn(
            batch_size, num_heads, seq_len, seq_len,
            dtype=torch.float32,
            device=self.device
        )
        
        expected = torch.nn.functional.softmax(input_torch, dim=-1)
        
        actual = self._run_cuda_kernel(
            compiled_kernel,
            inputs=[input_torch],
            output_shape=input_torch.shape,
            kernel_name="softmax_kernel",
            block_size=(256, 1, 1),
            grid_size=(seq_len, num_heads * batch_size, 1)
        )
        
        error_stats = self._compute_error_stats(expected, actual)
        
        return error_stats['max_abs_error'] < self.tolerance * 10, error_stats
    
    def _test_attention_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, float]]:
        """Test attention kernel correctness."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = min(model_info.get("max_seq_len", 512), 128)
        num_heads = model_info.get("num_attention_heads", 32)
        head_dim = model_info.get("hidden_size", 4096) // num_heads
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
        
        scale = 1.0 / np.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        expected = torch.matmul(attn_weights, v)
        
        actual = self._run_cuda_kernel(
            compiled_kernel,
            inputs=[q, k, v, torch.tensor(scale)],
            output_shape=expected.shape,
            kernel_name="attention_kernel",
            block_size=(128, 1, 1),
            grid_size=(num_heads, batch_size, 1)
        )
        
        error_stats = self._compute_error_stats(expected, actual)
        
        return error_stats['max_abs_error'] < self.tolerance * 100, error_stats
    
    def _run_cuda_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        inputs: List[torch.Tensor],
        output_shape: Tuple[int, ...],
        kernel_name: str,
        block_size: Tuple[int, int, int],
        grid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Run a CUDA kernel and return the output."""
        ptx_code = compiled_kernel.get("ptx_code", "")
        full_code = compiled_kernel.get("full_code", "")
        
        if not ptx_code and not full_code:
            raise ValueError("No compiled kernel code available")
        
        try:
            if full_code:
                mod = SourceModule(full_code)
            else:
                mod = cuda.module_from_buffer(ptx_code.encode())
            
            kernel_func = mod.get_function(kernel_name)
            
            gpu_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    gpu_inputs.append(cuda.mem_alloc(inp.nbytes))
                    cuda.memcpy_htod(gpu_inputs[-1], inp.cpu().numpy())
                else:
                    gpu_inputs.append(inp)
            
            output = torch.zeros(output_shape, dtype=torch.float32, device='cpu')
            gpu_output = cuda.mem_alloc(output.nbytes)
            
            kernel_func(
                *gpu_inputs,
                gpu_output,
                block=block_size,
                grid=grid_size
            )
            
            cuda.memcpy_dtoh(output.numpy(), gpu_output)
            
            return output.to(self.device)
            
        except Exception as e:
            console.print(f"[red]Error running CUDA kernel: {e}[/red]")
            return torch.zeros(output_shape, dtype=torch.float32, device=self.device)
    
    def _compute_error_stats(
        self,
        expected: torch.Tensor,
        actual: torch.Tensor
    ) -> Dict[str, float]:
        """Compute error statistics between expected and actual outputs."""
        diff = torch.abs(expected - actual)
        
        return {
            'max_abs_error': diff.max().item(),
            'mean_abs_error': diff.mean().item(),
            'relative_error': (diff / (torch.abs(expected) + 1e-8)).mean().item()
        }
    
    def generate_test_report(
        self,
        kernel_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive test report for multiple kernels."""
        report = {
            'total_kernels': len(kernel_results),
            'passed': sum(1 for r in kernel_results if r.get('passed', False)),
            'failed': sum(1 for r in kernel_results if not r.get('passed', False)),
            'results': kernel_results,
            'summary': {}
        }
        
        for result in kernel_results:
            op = result.get('operation', 'unknown')
            report['summary'][op] = {
                'passed': result.get('passed', False),
                'max_error': result.get('error_stats', {}).get('max_abs_error', float('inf'))
            }
        
        return report