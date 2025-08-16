import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import platform
import re
from rich.console import Console

console = Console()


class CUDACompiler:
    def __init__(self):
        self.nvcc_path = self._find_nvcc()
        if not self.nvcc_path:
            raise RuntimeError("NVIDIA CUDA Toolkit not found. Please install CUDA toolkit.")
        
        self.cuda_version = self._get_cuda_version()
        self.arch_flags = self._detect_gpu_architecture()
        
        console.print(f"[green]Found CUDA {self.cuda_version} at {self.nvcc_path}[/green]")
    
    def _find_nvcc(self) -> Optional[str]:
        """Find nvcc compiler in system PATH or common locations."""
        nvcc_name = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
        
        nvcc_path = shutil.which(nvcc_name)
        if nvcc_path:
            return nvcc_path
        
        common_paths = []
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin",
                r"C:\Program Files\NVIDIA Corporation\CUDA\v*\bin",
                r"C:\CUDA\v*\bin"
            ]
        else:
            common_paths = [
                "/usr/local/cuda/bin",
                "/usr/local/cuda-*/bin",
                "/opt/cuda/bin",
                "/opt/cuda-*/bin"
            ]
        
        import glob
        for pattern in common_paths:
            for path in glob.glob(pattern):
                nvcc_candidate = os.path.join(path, nvcc_name)
                if os.path.exists(nvcc_candidate):
                    return nvcc_candidate
        
        return None
    
    def _get_cuda_version(self) -> str:
        """Get CUDA toolkit version."""
        try:
            result = subprocess.run(
                [self.nvcc_path, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            
            version_match = re.search(r"release (\d+\.\d+)", result.stdout)
            if version_match:
                return version_match.group(1)
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _detect_gpu_architecture(self) -> List[str]:
        """Detect GPU compute capability and return appropriate arch flags."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            
            arch_flags = []
            for i in range(cuda.Device.count()):
                device = cuda.Device(i)
                major, minor = device.compute_capability()
                arch = f"{major}{minor}"
                arch_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
            
            return arch_flags
        except Exception:
            console.print("[yellow]Could not detect GPU architecture, using default sm_70[/yellow]")
            return ["-gencode=arch=compute_70,code=sm_70"]
    
    def compile_kernel(self, kernel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compile a CUDA kernel and return compilation results."""
        kernel_code = kernel_data.get("code", "")
        if not kernel_code:
            raise ValueError("No kernel code provided")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            cuda_file = temp_path / "kernel.cu"
            ptx_file = temp_path / "kernel.ptx"
            cubin_file = temp_path / "kernel.cubin"
            
            full_code = self._prepare_kernel_code(kernel_code)
            
            with open(cuda_file, 'w') as f:
                f.write(full_code)
            
            compile_flags = [
                "-ptx",
                "-O3",
                "-std=c++14",
                "-use_fast_math",
                "-lineinfo",
                f"-o={ptx_file}"
            ] + self.arch_flags
            
            constraints = kernel_data.get("constraints", {})
            if "max_registers" in constraints:
                compile_flags.append(f"-maxrregcount={constraints['max_registers']}")
            
            try:
                console.print(f"[cyan]Compiling kernel for {kernel_data.get('operation', 'unknown')}...[/cyan]")
                
                result = subprocess.run(
                    [self.nvcc_path] + compile_flags + [str(cuda_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                with open(ptx_file, 'r') as f:
                    ptx_code = f.read()
                
                cubin_result = subprocess.run(
                    [self.nvcc_path, "-cubin"] + self.arch_flags + ["-o", str(cubin_file), str(cuda_file)],
                    capture_output=True,
                    text=True
                )
                
                cubin_data = None
                if cubin_file.exists():
                    with open(cubin_file, 'rb') as f:
                        cubin_data = f.read()
                
                compilation_info = self._analyze_compilation_output(result.stderr)
                
                return {
                    **kernel_data,
                    "ptx_code": ptx_code,
                    "cubin_data": cubin_data,
                    "compilation_info": compilation_info,
                    "full_code": full_code,
                    "compile_success": True
                }
                
            except subprocess.CalledProcessError as e:
                error_info = self._parse_compilation_errors(e.stderr)
                raise RuntimeError(f"Kernel compilation failed:\n{error_info}")
    
    def _prepare_kernel_code(self, kernel_code: str) -> str:
        """Prepare kernel code with necessary includes and helpers."""
        includes = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, \
               cudaGetErrorString(error)); \
        return; \
    } \
} while(0)

#define FULL_MASK 0xffffffff

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

"""
        
        if "#include" not in kernel_code:
            return includes + kernel_code
        else:
            return kernel_code
    
    def _analyze_compilation_output(self, stderr: str) -> Dict[str, Any]:
        """Analyze nvcc compilation output for useful information."""
        info = {
            "warnings": [],
            "register_usage": {},
            "shared_memory": {},
            "spills": {}
        }
        
        warning_pattern = r"warning.*?:\s*(.*)"
        register_pattern = r"Function\s+(\w+).*?(\d+)\s+registers"
        shared_mem_pattern = r"Function\s+(\w+).*?(\d+)\s+bytes\s+smem"
        spill_pattern = r"Function\s+(\w+).*?(\d+)\s+bytes\s+spill"
        
        for match in re.finditer(warning_pattern, stderr):
            info["warnings"].append(match.group(1))
        
        for match in re.finditer(register_pattern, stderr):
            info["register_usage"][match.group(1)] = int(match.group(2))
        
        for match in re.finditer(shared_mem_pattern, stderr):
            info["shared_memory"][match.group(1)] = int(match.group(2))
        
        for match in re.finditer(spill_pattern, stderr):
            info["spills"][match.group(1)] = int(match.group(2))
        
        return info
    
    def _parse_compilation_errors(self, error_output: str) -> str:
        """Parse and format compilation errors."""
        lines = error_output.strip().split('\n')
        
        error_lines = []
        for line in lines:
            if "error:" in line or "Error:" in line:
                error_lines.append(line.strip())
            elif error_lines and line.strip():
                error_lines.append(f"  {line.strip()}")
        
        return '\n'.join(error_lines) if error_lines else error_output
    
    def get_kernel_info(self, compiled_kernel: Dict[str, Any]) -> Dict[str, Any]:
        """Extract kernel information from compiled data."""
        info = compiled_kernel.get("compilation_info", {})
        
        return {
            "operation": compiled_kernel.get("operation", "unknown"),
            "register_usage": info.get("register_usage", {}),
            "shared_memory": info.get("shared_memory", {}),
            "has_spills": bool(info.get("spills", {})),
            "warnings": info.get("warnings", []),
            "constraints": compiled_kernel.get("constraints", {})
        }
    
    def validate_kernel_launch_config(
        self,
        block_size: Tuple[int, int, int],
        grid_size: Tuple[int, int, int],
        shared_memory_bytes: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """Validate kernel launch configuration."""
        max_threads_per_block = 1024
        max_shared_memory = 48 * 1024
        max_grid_dim = 2147483647
        
        total_threads = block_size[0] * block_size[1] * block_size[2]
        if total_threads > max_threads_per_block:
            return False, f"Block size {total_threads} exceeds maximum {max_threads_per_block}"
        
        if shared_memory_bytes > max_shared_memory:
            return False, f"Shared memory {shared_memory_bytes} exceeds maximum {max_shared_memory}"
        
        for dim in grid_size:
            if dim > max_grid_dim:
                return False, f"Grid dimension {dim} exceeds maximum {max_grid_dim}"
        
        return True, None