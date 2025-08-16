import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
from typing import Dict, Any, List, Tuple, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import statistics

console = Console()


class Benchmarker:
    def __init__(self, iterations: int = 100, warmup_iterations: int = 10):
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot perform benchmarking.")
        
        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
    
    def __del__(self):
        """Clean up CUDA context."""
        if hasattr(self, 'cuda_context'):
            self.cuda_context.pop()
    
    def benchmark_kernel_standalone(
        self,
        compiled_kernel: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark a standalone user kernel."""
        kernel_name = analysis.get('kernel_name', 'user_kernel')
        
        console.print(f"\n[cyan]Benchmarking {kernel_name} kernel...[/cyan]")
        
        try:
            # Create synthetic test data based on kernel parameters
            test_data = self._create_test_data(analysis)
            
            # Run kernel with test data
            kernel_times = self._benchmark_cuda_kernel(
                compiled_kernel,
                inputs=test_data['inputs'],
                output_shape=test_data['output_shape'],
                kernel_name=kernel_name,
                block_size=test_data.get('block_size', (256, 1, 1)),
                grid_size=test_data.get('grid_size', (1024, 1, 1)),
                warmup=self.warmup_iterations,
                iterations=self.iterations
            )
            
            if kernel_times and all(t != float('inf') for t in kernel_times):
                return {
                    "avg_time_ms": statistics.median(kernel_times),
                    "min_time_ms": min(kernel_times),
                    "max_time_ms": max(kernel_times),
                    "std_dev_ms": statistics.stdev(kernel_times) if len(kernel_times) > 1 else 0,
                    "success": True
                }
            else:
                return {
                    "avg_time_ms": float('inf'),
                    "min_time_ms": float('inf'),
                    "max_time_ms": float('inf'),
                    "std_dev_ms": 0,
                    "success": False
                }
                
        except Exception as e:
            console.print(f"[red]Error during benchmarking: {e}[/red]")
            return {
                "avg_time_ms": float('inf'),
                "min_time_ms": float('inf'),
                "max_time_ms": float('inf'),
                "std_dev_ms": 0,
                "success": False,
                "error": str(e)
            }
    
    def _create_test_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create synthetic test data based on kernel analysis."""
        # Default test data size
        size = 1024 * 1024  # 1M elements
        
        # Adjust based on detected patterns
        if "matrix multiplication" in analysis.get('patterns', []):
            m, n, k = 1024, 1024, 1024
            return {
                'inputs': [
                    torch.randn(m, k, dtype=torch.float32, device=self.device),
                    torch.randn(k, n, dtype=torch.float32, device=self.device)
                ],
                'output_shape': (m, n),
                'block_size': (16, 16, 1),
                'grid_size': ((n + 15) // 16, (m + 15) // 16, 1)
            }
        elif "reduction" in analysis.get('patterns', []):
            return {
                'inputs': [torch.randn(size, dtype=torch.float32, device=self.device)],
                'output_shape': (1,),
                'block_size': (256, 1, 1),
                'grid_size': ((size + 255) // 256, 1, 1)
            }
        else:
            # Generic test data
            return {
                'inputs': [torch.randn(size, dtype=torch.float32, device=self.device)],
                'output_shape': (size,),
                'block_size': (256, 1, 1),
                'grid_size': ((size + 255) // 256, 1, 1)
            }
    
    def benchmark_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark a compiled kernel and compare with PyTorch baseline."""
        operation = compiled_kernel.get("operation", "unknown")
        
        console.print(f"\n[cyan]Benchmarking {operation} kernel...[/cyan]")
        
        benchmark_functions = {
            "matmul": self._benchmark_matmul,
            "layernorm": self._benchmark_layernorm,
            "gelu": self._benchmark_gelu,
            "softmax": self._benchmark_softmax,
            "attention": self._benchmark_attention
        }
        
        if operation not in benchmark_functions:
            console.print(f"[yellow]No benchmark available for {operation}[/yellow]")
            return {"avg_time_ms": float('inf'), "baseline_time_ms": float('inf')}
        
        try:
            benchmark_func = benchmark_functions[operation]
            results = benchmark_func(compiled_kernel, model_info)
            
            speedup = results['baseline_time_ms'] / results['kernel_time_ms']
            results['speedup'] = speedup
            
            console.print(f"[green]Benchmark results for {operation}:[/green]")
            console.print(f"  Baseline (PyTorch): {results['baseline_time_ms']:.3f} ms")
            console.print(f"  Optimized kernel: {results['kernel_time_ms']:.3f} ms")
            console.print(f"  Speedup: {speedup:.2f}x")
            console.print(f"  Throughput: {results.get('throughput_gbps', 0):.2f} GB/s")
            
            return {
                "avg_time_ms": results['kernel_time_ms'],
                "baseline_time_ms": results['baseline_time_ms'],
                "speedup": speedup,
                "throughput_gbps": results.get('throughput_gbps', 0),
                "full_results": results
            }
            
        except Exception as e:
            console.print(f"[red]Error during benchmarking: {e}[/red]")
            return {"avg_time_ms": float('inf'), "baseline_time_ms": float('inf')}
    
    def _benchmark_matmul(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark matrix multiplication kernel."""
        batch_size = model_info.get("batch_size", 1)
        hidden_size = model_info.get("hidden_size", 4096)
        
        m, k, n = batch_size, hidden_size, hidden_size
        
        a_torch = torch.randn(m, k, dtype=torch.float32, device=self.device)
        b_torch = torch.randn(k, n, dtype=torch.float32, device=self.device)
        c_torch = torch.zeros(m, n, dtype=torch.float32, device=self.device)
        
        baseline_times = self._benchmark_pytorch_op(
            lambda: torch.matmul(a_torch, b_torch),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        kernel_times = self._benchmark_cuda_kernel(
            compiled_kernel,
            inputs=[a_torch, b_torch, c_torch],
            output_shape=(m, n),
            kernel_name="matmul_kernel",
            block_size=(16, 16, 1),
            grid_size=((n + 15) // 16, (m + 15) // 16, 1),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        total_bytes = (m * k + k * n + m * n) * 4
        throughput = (total_bytes / 1e9) / (min(kernel_times) / 1000)
        
        return {
            "baseline_time_ms": statistics.median(baseline_times),
            "kernel_time_ms": statistics.median(kernel_times),
            "baseline_std_ms": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "kernel_std_ms": statistics.stdev(kernel_times) if len(kernel_times) > 1 else 0,
            "throughput_gbps": throughput,
            "problem_size": f"{m}x{k}x{n}"
        }
    
    def _benchmark_layernorm(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark layer normalization kernel."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        hidden_size = model_info.get("hidden_size", 4096)
        
        input_torch = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=self.device)
        gamma = torch.ones(hidden_size, dtype=torch.float32, device=self.device)
        beta = torch.zeros(hidden_size, dtype=torch.float32, device=self.device)
        eps = 1e-5
        
        baseline_times = self._benchmark_pytorch_op(
            lambda: torch.nn.functional.layer_norm(input_torch, (hidden_size,), gamma, beta, eps),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        kernel_times = self._benchmark_cuda_kernel(
            compiled_kernel,
            inputs=[input_torch, gamma, beta, torch.tensor(eps)],
            output_shape=(batch_size, seq_len, hidden_size),
            kernel_name="layernorm_kernel",
            block_size=(256, 1, 1),
            grid_size=(seq_len, batch_size, 1),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        total_bytes = batch_size * seq_len * hidden_size * 4 * 3
        throughput = (total_bytes / 1e9) / (min(kernel_times) / 1000)
        
        return {
            "baseline_time_ms": statistics.median(baseline_times),
            "kernel_time_ms": statistics.median(kernel_times),
            "baseline_std_ms": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "kernel_std_ms": statistics.stdev(kernel_times) if len(kernel_times) > 1 else 0,
            "throughput_gbps": throughput,
            "problem_size": f"{batch_size}x{seq_len}x{hidden_size}"
        }
    
    def _benchmark_gelu(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark GELU activation kernel."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        hidden_size = model_info.get("hidden_size", 4096)
        
        total_elements = batch_size * seq_len * hidden_size
        input_torch = torch.randn(total_elements, dtype=torch.float32, device=self.device)
        
        baseline_times = self._benchmark_pytorch_op(
            lambda: torch.nn.functional.gelu(input_torch),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        kernel_times = self._benchmark_cuda_kernel(
            compiled_kernel,
            inputs=[input_torch],
            output_shape=input_torch.shape,
            kernel_name="gelu_kernel",
            block_size=(256, 1, 1),
            grid_size=((total_elements + 255) // 256, 1, 1),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        total_bytes = total_elements * 4 * 2
        throughput = (total_bytes / 1e9) / (min(kernel_times) / 1000)
        
        return {
            "baseline_time_ms": statistics.median(baseline_times),
            "kernel_time_ms": statistics.median(kernel_times),
            "baseline_std_ms": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "kernel_std_ms": statistics.stdev(kernel_times) if len(kernel_times) > 1 else 0,
            "throughput_gbps": throughput,
            "problem_size": f"{total_elements} elements"
        }
    
    def _benchmark_softmax(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark softmax kernel."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        num_heads = model_info.get("num_attention_heads", 32)
        
        input_torch = torch.randn(
            batch_size, num_heads, seq_len, seq_len,
            dtype=torch.float32,
            device=self.device
        )
        
        baseline_times = self._benchmark_pytorch_op(
            lambda: torch.nn.functional.softmax(input_torch, dim=-1),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        kernel_times = self._benchmark_cuda_kernel(
            compiled_kernel,
            inputs=[input_torch],
            output_shape=input_torch.shape,
            kernel_name="softmax_kernel",
            block_size=(256, 1, 1),
            grid_size=(seq_len, num_heads * batch_size, 1),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        total_bytes = batch_size * num_heads * seq_len * seq_len * 4 * 2
        throughput = (total_bytes / 1e9) / (min(kernel_times) / 1000)
        
        return {
            "baseline_time_ms": statistics.median(baseline_times),
            "kernel_time_ms": statistics.median(kernel_times),
            "baseline_std_ms": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "kernel_std_ms": statistics.stdev(kernel_times) if len(kernel_times) > 1 else 0,
            "throughput_gbps": throughput,
            "problem_size": f"{batch_size}x{num_heads}x{seq_len}x{seq_len}"
        }
    
    def _benchmark_attention(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark attention kernel."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = min(model_info.get("max_seq_len", 512), 128)
        num_heads = model_info.get("num_attention_heads", 32)
        head_dim = model_info.get("hidden_size", 4096) // num_heads
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
        scale = 1.0 / np.sqrt(head_dim)
        
        def pytorch_attention():
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, v)
        
        baseline_times = self._benchmark_pytorch_op(
            pytorch_attention,
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        kernel_times = self._benchmark_cuda_kernel(
            compiled_kernel,
            inputs=[q, k, v, torch.tensor(scale)],
            output_shape=(batch_size, num_heads, seq_len, head_dim),
            kernel_name="attention_kernel",
            block_size=(128, 1, 1),
            grid_size=(num_heads, batch_size, 1),
            warmup=self.warmup_iterations,
            iterations=self.iterations
        )
        
        flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        tflops = (flops / 1e12) / (min(kernel_times) / 1000)
        
        return {
            "baseline_time_ms": statistics.median(baseline_times),
            "kernel_time_ms": statistics.median(kernel_times),
            "baseline_std_ms": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "kernel_std_ms": statistics.stdev(kernel_times) if len(kernel_times) > 1 else 0,
            "tflops": tflops,
            "problem_size": f"B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}"
        }
    
    def _benchmark_pytorch_op(
        self,
        op_func,
        warmup: int = 10,
        iterations: int = 100
    ) -> List[float]:
        """Benchmark a PyTorch operation."""
        torch.cuda.synchronize()
        
        for _ in range(warmup):
            op_func()
        torch.cuda.synchronize()
        
        times = []
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("PyTorch baseline", total=iterations)
            
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                op_func()
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
                progress.advance(task)
        
        return times
    
    def _benchmark_cuda_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        inputs: List[torch.Tensor],
        output_shape: Tuple[int, ...],
        kernel_name: str,
        block_size: Tuple[int, int, int],
        grid_size: Tuple[int, int, int],
        warmup: int = 10,
        iterations: int = 100
    ) -> List[float]:
        """Benchmark a CUDA kernel."""
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
            
            for _ in range(warmup):
                kernel_func(
                    *gpu_inputs,
                    gpu_output,
                    block=block_size,
                    grid=grid_size
                )
            cuda.Context.synchronize()
            
            times = []
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("CUDA kernel", total=iterations)
                
                for _ in range(iterations):
                    start_event = cuda.Event()
                    end_event = cuda.Event()
                    
                    start_event.record()
                    kernel_func(
                        *gpu_inputs,
                        gpu_output,
                        block=block_size,
                        grid=grid_size
                    )
                    end_event.record()
                    end_event.synchronize()
                    
                    elapsed_ms = start_event.time_till(end_event)
                    times.append(elapsed_ms)
                    progress.advance(task)
            
            return times
            
        except Exception as e:
            console.print(f"[red]Error running CUDA kernel benchmark: {e}[/red]")
            return [float('inf')] * iterations
    
    def generate_benchmark_report(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        report = {
            "summary": {
                "total_kernels": len(results),
                "average_speedup": statistics.mean([r.get("speedup", 1.0) for r in results]),
                "best_speedup": max([r.get("speedup", 1.0) for r in results]),
                "worst_speedup": min([r.get("speedup", 1.0) for r in results])
            },
            "detailed_results": results
        }
        
        return report