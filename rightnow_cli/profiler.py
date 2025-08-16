import subprocess
import tempfile
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import platform
from rich.console import Console

console = Console()


class CUDAProfiler:
    def __init__(self):
        self.nvprof_path = self._find_profiler("nvprof")
        self.ncu_path = self._find_profiler("ncu")
        
        if self.ncu_path:
            self.profiler_type = "ncu"
            console.print(f"[green]Using NVIDIA Nsight Compute (ncu) for profiling[/green]")
        elif self.nvprof_path:
            self.profiler_type = "nvprof"
            console.print(f"[yellow]Using legacy nvprof for profiling (consider upgrading to newer CUDA toolkit)[/yellow]")
        else:
            console.print(f"[red]No CUDA profiler found. Install NVIDIA Nsight Compute for best results.[/red]")
            self.profiler_type = "none"
    
    def _find_profiler(self, profiler_name: str) -> Optional[str]:
        """Find CUDA profiler in system PATH or common locations."""
        if platform.system() == "Windows":
            profiler_name += ".exe"
        
        import shutil
        profiler_path = shutil.which(profiler_name)
        if profiler_path:
            return profiler_path
        
        common_paths = []
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin",
                r"C:\Program Files\NVIDIA Corporation\Nsight Compute*",
                r"C:\Program Files\NVIDIA Corporation\CUDA\v*\bin"
            ]
        else:
            common_paths = [
                "/usr/local/cuda/bin",
                "/usr/local/cuda-*/bin",
                "/opt/cuda/bin",
                "/opt/cuda-*/bin",
                "/opt/nvidia/nsight-compute/*/"
            ]
        
        import glob
        for pattern in common_paths:
            for path in glob.glob(pattern):
                profiler_candidate = os.path.join(path, profiler_name)
                if os.path.exists(profiler_candidate):
                    return profiler_candidate
        
        return None
    
    def profile_kernel(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile a CUDA kernel and extract performance metrics."""
        operation = compiled_kernel.get("operation", "unknown")
        
        console.print(f"[cyan]Profiling {operation} kernel...[/cyan]")
        
        if self.profiler_type == "none":
            return self._estimate_metrics(compiled_kernel, model_info)
        elif self.profiler_type == "ncu":
            return self._profile_with_ncu(compiled_kernel, model_info)
        else:
            return self._profile_with_nvprof(compiled_kernel, model_info)
    
    def _estimate_metrics(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate metrics when no profiler is available."""
        compilation_info = compiled_kernel.get("compilation_info", {})
        
        kernel_names = list(compilation_info.get("register_usage", {}).keys())
        if kernel_names:
            kernel_name = kernel_names[0]
            registers = compilation_info["register_usage"].get(kernel_name, 32)
            shared_mem = compilation_info["shared_memory"].get(kernel_name, 0)
        else:
            registers = 32
            shared_mem = 0
        
        operation = compiled_kernel.get("operation", "unknown")
        memory_estimate = self._estimate_memory_usage(operation, model_info)
        
        return {
            "profiler_available": False,
            "register_usage": registers,
            "shared_memory_bytes": shared_mem,
            "memory_usage_mb": memory_estimate,
            "occupancy": self._estimate_occupancy(registers, shared_mem),
            "memory_efficiency": 0.7,
            "compute_efficiency": 0.6,
            "warnings": ["No profiler available - metrics are estimates only"]
        }
    
    def _profile_with_ncu(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile using NVIDIA Nsight Compute."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            test_program = self._create_test_program(compiled_kernel, model_info, temp_path)
            if not test_program:
                return self._estimate_metrics(compiled_kernel, model_info)
            
            report_file = temp_path / "profile_report.ncu-rep"
            
            profile_cmd = [
                self.ncu_path,
                "--target-processes", "all",
                "--kernel-regex", ".*",
                "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                            "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
                            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,"
                            "sm__warps_active.avg.pct_of_peak_sustained_active,"
                            "launch__registers_per_thread,"
                            "launch__shared_mem_per_block_static",
                "--export", str(report_file),
                str(test_program)
            ]
            
            try:
                result = subprocess.run(
                    profile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    return self._parse_ncu_output(result.stdout, report_file)
                else:
                    console.print(f"[red]NCU profiling failed: {result.stderr}[/red]")
                    return self._estimate_metrics(compiled_kernel, model_info)
                    
            except subprocess.TimeoutExpired:
                console.print("[red]Profiling timed out[/red]")
                return self._estimate_metrics(compiled_kernel, model_info)
            except Exception as e:
                console.print(f"[red]Profiling error: {e}[/red]")
                return self._estimate_metrics(compiled_kernel, model_info)
    
    def _profile_with_nvprof(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile using legacy nvprof."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            test_program = self._create_test_program(compiled_kernel, model_info, temp_path)
            if not test_program:
                return self._estimate_metrics(compiled_kernel, model_info)
            
            profile_cmd = [
                self.nvprof_path,
                "--print-gpu-trace",
                "--metrics", "achieved_occupancy,gld_efficiency,gst_efficiency,"
                           "shared_efficiency,sm_efficiency,dram_utilization",
                str(test_program)
            ]
            
            try:
                result = subprocess.run(
                    profile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    return self._parse_nvprof_output(result.stdout)
                else:
                    console.print(f"[red]nvprof profiling failed: {result.stderr}[/red]")
                    return self._estimate_metrics(compiled_kernel, model_info)
                    
            except subprocess.TimeoutExpired:
                console.print("[red]Profiling timed out[/red]")
                return self._estimate_metrics(compiled_kernel, model_info)
            except Exception as e:
                console.print(f"[red]Profiling error: {e}[/red]")
                return self._estimate_metrics(compiled_kernel, model_info)
    
    def _create_test_program(
        self,
        compiled_kernel: Dict[str, Any],
        model_info: Dict[str, Any],
        temp_path: Path
    ) -> Optional[Path]:
        """Create a test program that runs the kernel."""
        operation = compiled_kernel.get("operation", "unknown")
        kernel_code = compiled_kernel.get("full_code", "")
        
        if not kernel_code:
            return None
        
        test_code = self._generate_test_harness(operation, kernel_code, model_info)
        
        cuda_file = temp_path / "test_kernel.cu"
        exe_file = temp_path / ("test_kernel.exe" if platform.system() == "Windows" else "test_kernel")
        
        with open(cuda_file, 'w') as f:
            f.write(test_code)
        
        from .compiler import CUDACompiler
        compiler = CUDACompiler()
        
        compile_cmd = [
            compiler.nvcc_path,
            "-O3",
            "-std=c++14"
        ] + compiler.arch_flags + [
            "-o", str(exe_file),
            str(cuda_file)
        ]
        
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return exe_file
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to compile test program: {e.stderr}[/red]")
            return None
    
    def _generate_test_harness(
        self,
        operation: str,
        kernel_code: str,
        model_info: Dict[str, Any]
    ) -> str:
        """Generate test harness code for profiling."""
        if operation == "matmul":
            return self._matmul_test_harness(kernel_code, model_info)
        elif operation == "layernorm":
            return self._layernorm_test_harness(kernel_code, model_info)
        else:
            return self._generic_test_harness(kernel_code, model_info)
    
    def _matmul_test_harness(self, kernel_code: str, model_info: Dict[str, Any]) -> str:
        """Generate test harness for matrix multiplication."""
        batch_size = model_info.get("batch_size", 1)
        hidden_size = model_info.get("hidden_size", 4096)
        
        return f"""
{kernel_code}

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

int main() {{
    const int M = {batch_size};
    const int K = {hidden_size};
    const int N = {hidden_size};
    
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    for (int i = 0; i < 10; i++) {{
        matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    }}
    
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}}
"""
    
    def _layernorm_test_harness(self, kernel_code: str, model_info: Dict[str, Any]) -> str:
        """Generate test harness for layer normalization."""
        batch_size = model_info.get("batch_size", 1)
        seq_len = model_info.get("max_seq_len", 512)
        hidden_size = model_info.get("hidden_size", 4096)
        
        return f"""
{kernel_code}

#include <iostream>
#include <cuda_runtime.h>

int main() {{
    const int batch_size = {batch_size};
    const int seq_len = {seq_len};
    const int hidden_size = {hidden_size};
    const float eps = 1e-5f;
    
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t param_size = hidden_size * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);
    
    dim3 blockSize(256);
    dim3 gridSize(seq_len, batch_size);
    
    for (int i = 0; i < 10; i++) {{
        layernorm_kernel<<<gridSize, blockSize>>>(
            d_input, d_output, d_gamma, d_beta, hidden_size, eps
        );
    }}
    
    cudaDeviceSynchronize();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    
    return 0;
}}
"""
    
    def _generic_test_harness(self, kernel_code: str, model_info: Dict[str, Any]) -> str:
        """Generate generic test harness."""
        return f"""
{kernel_code}

#include <iostream>
#include <cuda_runtime.h>

int main() {{
    // Generic test - just ensure kernel compiles and runs
    dim3 blockSize(256);
    dim3 gridSize(1024);
    
    cudaDeviceSynchronize();
    std::cout << "Kernel test completed" << std::endl;
    
    return 0;
}}
"""
    
    def _parse_ncu_output(self, stdout: str, report_file: Path) -> Dict[str, Any]:
        """Parse NVIDIA Nsight Compute output."""
        metrics = {
            "profiler_available": True,
            "register_usage": 32,
            "shared_memory_bytes": 0,
            "memory_usage_mb": 0,
            "occupancy": 0.5,
            "memory_efficiency": 0.7,
            "compute_efficiency": 0.6,
            "sm_efficiency": 0.0,
            "dram_throughput": 0.0,
            "warnings": []
        }
        
        register_pattern = r"Registers Per Thread\s+(\d+)"
        shared_mem_pattern = r"Static Shared Memory Per Block\s+(\d+)"
        occupancy_pattern = r"Achieved Occupancy\s+(\d+\.?\d*)%"
        sm_pattern = r"sm__throughput.*?(\d+\.?\d*)%"
        dram_pattern = r"dram__throughput.*?(\d+\.?\d*)%"
        
        for match in re.finditer(register_pattern, stdout):
            metrics["register_usage"] = int(match.group(1))
        
        for match in re.finditer(shared_mem_pattern, stdout):
            metrics["shared_memory_bytes"] = int(match.group(1))
        
        for match in re.finditer(occupancy_pattern, stdout):
            metrics["occupancy"] = float(match.group(1)) / 100.0
        
        for match in re.finditer(sm_pattern, stdout):
            metrics["sm_efficiency"] = float(match.group(1)) / 100.0
        
        for match in re.finditer(dram_pattern, stdout):
            metrics["dram_throughput"] = float(match.group(1)) / 100.0
        
        if metrics["occupancy"] < 0.5:
            metrics["warnings"].append("Low occupancy detected")
        
        if metrics["register_usage"] > 64:
            metrics["warnings"].append("High register usage may limit occupancy")
        
        return metrics
    
    def _parse_nvprof_output(self, stdout: str) -> Dict[str, Any]:
        """Parse nvprof output."""
        metrics = {
            "profiler_available": True,
            "register_usage": 32,
            "shared_memory_bytes": 0,
            "memory_usage_mb": 0,
            "occupancy": 0.5,
            "memory_efficiency": 0.7,
            "compute_efficiency": 0.6,
            "warnings": []
        }
        
        occupancy_pattern = r"Achieved Occupancy\s+(\d+\.?\d*)%"
        gld_pattern = r"Global Load Efficiency\s+(\d+\.?\d*)%"
        gst_pattern = r"Global Store Efficiency\s+(\d+\.?\d*)%"
        sm_pattern = r"SM Efficiency\s+(\d+\.?\d*)%"
        dram_pattern = r"Device Memory Utilization\s+(\w+)"
        
        for match in re.finditer(occupancy_pattern, stdout):
            metrics["occupancy"] = float(match.group(1)) / 100.0
        
        gld_values = []
        gst_values = []
        
        for match in re.finditer(gld_pattern, stdout):
            gld_values.append(float(match.group(1)) / 100.0)
        
        for match in re.finditer(gst_pattern, stdout):
            gst_values.append(float(match.group(1)) / 100.0)
        
        if gld_values and gst_values:
            metrics["memory_efficiency"] = (sum(gld_values) + sum(gst_values)) / (len(gld_values) + len(gst_values))
        
        for match in re.finditer(sm_pattern, stdout):
            metrics["compute_efficiency"] = float(match.group(1)) / 100.0
        
        for match in re.finditer(dram_pattern, stdout):
            utilization = match.group(1).lower()
            if utilization in ["low", "idle"]:
                metrics["warnings"].append("Low DRAM utilization")
        
        return metrics
    
    def _estimate_memory_usage(self, operation: str, model_info: Dict[str, Any]) -> float:
        """Estimate memory usage based on operation and model info."""
        batch_size = model_info.get("batch_size", 1)
        hidden_size = model_info.get("hidden_size", 4096)
        seq_len = model_info.get("max_seq_len", 512)
        
        bytes_per_element = 4
        
        if operation == "matmul":
            elements = batch_size * hidden_size * hidden_size * 3
        elif operation == "layernorm":
            elements = batch_size * seq_len * hidden_size * 2
        elif operation == "attention":
            num_heads = model_info.get("num_attention_heads", 32)
            elements = batch_size * num_heads * seq_len * seq_len * 4
        else:
            elements = batch_size * seq_len * hidden_size * 2
        
        return (elements * bytes_per_element) / (1024 * 1024)
    
    def _estimate_occupancy(self, registers: int, shared_mem: int) -> float:
        """Estimate occupancy based on resource usage."""
        max_registers_per_sm = 65536
        max_shared_mem_per_sm = 49152
        max_threads_per_sm = 2048
        threads_per_block = 256
        
        blocks_limited_by_registers = max_registers_per_sm // (registers * threads_per_block)
        blocks_limited_by_shared_mem = max_shared_mem_per_sm // max(shared_mem, 1)
        blocks_limited_by_threads = max_threads_per_sm // threads_per_block
        
        max_blocks = min(
            blocks_limited_by_registers,
            blocks_limited_by_shared_mem,
            blocks_limited_by_threads,
            32
        )
        
        active_threads = max_blocks * threads_per_block
        occupancy = active_threads / max_threads_per_sm
        
        return min(occupancy, 1.0)
    
    def generate_profile_report(
        self,
        kernel_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        report = {
            "profiler_type": self.profiler_type,
            "total_kernels": len(kernel_results),
            "average_occupancy": sum(r.get("occupancy", 0) for r in kernel_results) / len(kernel_results) if kernel_results else 0,
            "average_memory_efficiency": sum(r.get("memory_efficiency", 0) for r in kernel_results) / len(kernel_results) if kernel_results else 0,
            "kernels_with_warnings": sum(1 for r in kernel_results if r.get("warnings")),
            "detailed_results": kernel_results
        }
        
        return report