import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class KernelAnalyzer:
    """Analyzes CUDA kernels to extract information and identify optimization opportunities."""
    
    def analyze_kernel(self, code: str) -> Dict[str, Any]:
        """
        Analyze a CUDA kernel and extract relevant information.
        
        Returns:
            Dictionary containing:
            - kernel_name: Name of the kernel function
            - parameters: List of parameter names
            - shared_memory_usage: Detected shared memory usage
            - patterns: Detected computational patterns
            - optimization_opportunities: List of potential optimizations
            - performance_hints: Performance-related suggestions
        """
        analysis = {
            "kernel_name": self._extract_kernel_name(code),
            "parameters": self._extract_parameters(code),
            "launch_bounds": self._extract_launch_bounds(code),
            "shared_memory_usage": self._analyze_shared_memory(code),
            "global_accesses": self._analyze_global_memory(code),
            "patterns": self._detect_patterns(code),
            "optimization_opportunities": [],
            "performance_hints": [],
            "complexity": self._estimate_complexity(code),
            "synchronization": self._analyze_synchronization(code),
            "arithmetic_intensity": self._estimate_arithmetic_intensity(code)
        }
        
        # Identify optimization opportunities
        analysis["optimization_opportunities"] = self._identify_optimizations(code, analysis)
        analysis["performance_hints"] = self._generate_performance_hints(code, analysis)
        
        return analysis
    
    def _extract_kernel_name(self, code: str) -> str:
        """Extract the kernel function name."""
        kernel_pattern = r'__global__\s+\w+\s+(\w+)\s*\('
        match = re.search(kernel_pattern, code)
        return match.group(1) if match else "unknown_kernel"
    
    def _extract_parameters(self, code: str) -> List[str]:
        """Extract kernel parameters."""
        kernel_pattern = r'__global__\s+\w+\s+\w+\s*\(([^)]+)\)'
        match = re.search(kernel_pattern, code)
        if not match:
            return []
        
        params_str = match.group(1)
        params = []
        
        # Simple parameter parsing
        for param in params_str.split(','):
            param = param.strip()
            # Extract parameter name (last word)
            words = param.split()
            if words:
                param_name = words[-1].strip('*&')
                params.append(param_name)
        
        return params
    
    def _extract_launch_bounds(self, code: str) -> Optional[str]:
        """Extract launch bounds if specified."""
        bounds_pattern = r'__launch_bounds__\s*\(([^)]+)\)'
        match = re.search(bounds_pattern, code)
        return match.group(1) if match else None
    
    def _analyze_shared_memory(self, code: str) -> str:
        """Analyze shared memory usage."""
        shared_patterns = [
            r'__shared__\s+(\w+)\s+(\w+)\[([^\]]+)\]',  # Static shared memory
            r'extern\s+__shared__\s+(\w+)\s+(\w+)\[\]',  # Dynamic shared memory
        ]
        
        total_bytes = 0
        shared_vars = []
        
        for pattern in shared_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if len(match) >= 3:
                    type_name = match[0]
                    var_name = match[1]
                    size_expr = match[2] if len(match) > 2 else "dynamic"
                    
                    # Estimate size
                    type_sizes = {
                        'float': 4, 'int': 4, 'double': 8, 
                        'char': 1, 'short': 2, 'long': 8
                    }
                    
                    base_size = type_sizes.get(type_name, 4)
                    
                    if size_expr != "dynamic":
                        try:
                            # Simple size calculation
                            size_val = eval(size_expr.replace('BLOCK_SIZE', '256').replace('TILE_SIZE', '16'))
                            total_bytes += base_size * size_val
                        except:
                            pass
                    
                    shared_vars.append(f"{var_name} ({type_name}[{size_expr}])")
        
        if shared_vars:
            return f"{total_bytes} bytes ({', '.join(shared_vars)})"
        return "None detected"
    
    def _analyze_global_memory(self, code: str) -> Dict[str, Any]:
        """Analyze global memory access patterns."""
        global_reads = len(re.findall(r'\w+\[[^\]]+\](?!\s*=)', code))
        global_writes = len(re.findall(r'\w+\[[^\]]+\]\s*=', code))
        
        # Check for coalesced access patterns
        coalesced_patterns = [
            r'threadIdx\.x',
            r'blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x',
            r'tid|thread_id|idx'
        ]
        
        has_coalesced = any(re.search(pattern, code) for pattern in coalesced_patterns)
        
        return {
            "reads": global_reads,
            "writes": global_writes,
            "likely_coalesced": has_coalesced
        }
    
    def _detect_patterns(self, code: str) -> List[str]:
        """Detect common computational patterns."""
        patterns = []
        
        pattern_checks = {
            "reduction": [r'__syncthreads', r'for.*>>=\s*1', r'shared\[.*threadIdx'],
            "tiling": [r'TILE_SIZE', r'tile', r'__shared__.*\[.*\]\[.*\]'],
            "matrix multiplication": [r'\.x\].*\[.*\.y', r'row.*col', r'\.y\].*\[.*\.x'],
            "stencil": [r'[-+]\s*1\]', r'[+-]\s*blockDim', r'neighbor'],
            "scan/prefix sum": [r'inclusive.*scan', r'exclusive.*scan', r'prefix'],
            "transpose": [r'\.y\]\[.*\.x', r'\.x\]\[.*\.y'],
            "convolution": [r'filter', r'kernel.*\[.*\]', r'conv'],
            "parallel reduction": [r'atomicAdd', r'warp.*reduce', r'__shfl']
        }
        
        for pattern_name, indicators in pattern_checks.items():
            if sum(1 for ind in indicators if re.search(ind, code, re.IGNORECASE)) >= 2:
                patterns.append(pattern_name)
        
        return patterns
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate computational complexity."""
        loop_count = len(re.findall(r'\bfor\s*\(', code))
        while_count = len(re.findall(r'\bwhile\s*\(', code))
        
        if loop_count >= 3 or while_count >= 2:
            return "High"
        elif loop_count >= 1 or while_count >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_synchronization(self, code: str) -> Dict[str, int]:
        """Analyze synchronization usage."""
        return {
            "syncthreads": len(re.findall(r'__syncthreads\(\)', code)),
            "syncwarp": len(re.findall(r'__syncwarp\(\)', code)),
            "atomic_ops": len(re.findall(r'atomic\w+\(', code)),
            "barriers": len(re.findall(r'__threadfence|bar\.sync', code))
        }
    
    def _estimate_arithmetic_intensity(self, code: str) -> float:
        """Estimate arithmetic intensity (ops per memory access)."""
        # Count arithmetic operations
        arith_ops = len(re.findall(r'[+\-*/](?!=)', code))
        math_funcs = len(re.findall(r'(sin|cos|exp|log|sqrt|pow|fma)\s*\(', code))
        total_ops = arith_ops + math_funcs * 10  # Math functions count as more ops
        
        # Count memory accesses
        mem_pattern = r'\w+\[[^\]]+\]'
        mem_accesses = len(re.findall(mem_pattern, code))
        
        if mem_accesses > 0:
            return total_ops / mem_accesses
        return 0.0
    
    def _identify_optimizations(self, code: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential optimization opportunities."""
        opportunities = []
        
        # Check for uncoalesced memory access
        if not analysis["global_accesses"].get("likely_coalesced"):
            opportunities.append("Consider coalescing global memory accesses using threadIdx.x")
        
        # Check for shared memory bank conflicts
        if "__shared__" in code and not re.search(r'\[\w+\s*\+\s*1\]', code):
            opportunities.append("Add padding to shared memory arrays to avoid bank conflicts")
        
        # Check for missing loop unrolling
        if "for" in code and "#pragma unroll" not in code:
            opportunities.append("Consider loop unrolling with #pragma unroll")
        
        # Check for missing vectorization
        if not re.search(r'(float2|float4|int2|int4|double2)', code):
            opportunities.append("Consider using vector types (float2/float4) for memory access")
        
        # Check for inefficient synchronization
        sync_data = analysis["synchronization"]
        if sync_data["syncthreads"] > 3:
            opportunities.append("High syncthreads count - consider reducing synchronization")
        
        # Check for register spilling potential
        if code.count("float") + code.count("int") > 50:
            opportunities.append("High variable count - watch for register spilling")
        
        # Check arithmetic intensity
        if analysis["arithmetic_intensity"] < 1.0:
            opportunities.append("Low arithmetic intensity - consider fusing operations")
        
        # Pattern-specific optimizations
        if "reduction" in analysis["patterns"] and "warp" not in code:
            opportunities.append("Consider warp-level primitives for reduction")
        
        if "matrix multiplication" in analysis["patterns"] and "tensor" not in code.lower():
            opportunities.append("Consider Tensor Core operations for matrix multiplication")
        
        return opportunities
    
    def _generate_performance_hints(self, code: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance hints based on analysis."""
        hints = []
        
        # Memory access hints
        if analysis["global_accesses"]["reads"] > 20:
            hints.append("High global memory reads - consider caching in shared memory")
        
        # Divergence hints
        if re.search(r'if.*threadIdx', code):
            hints.append("Thread divergence detected - minimize conditional execution")
        
        # Atomic operation hints
        if analysis["synchronization"]["atomic_ops"] > 0:
            hints.append("Atomic operations can be bottlenecks - consider alternatives")
        
        # Occupancy hints
        if analysis.get("shared_memory_usage", "").startswith("0"):
            hints.append("No shared memory usage - could improve data reuse")
        
        # Math optimization hints
        if re.search(r'/(float|double)', code):
            hints.append("Division operations are expensive - consider reciprocal multiplication")
        
        if re.search(r'sin|cos|exp|log', code) and "fast_math" not in code:
            hints.append("Consider -use_fast_math for transcendental functions")
        
        return hints
    
    def extract_test_data(self, code: str) -> Optional[Dict[str, Any]]:
        """Extract test data from comments if available."""
        test_pattern = r'//\s*TEST:\s*(.+)'
        matches = re.findall(test_pattern, code)
        
        if matches:
            # Parse simple test data format
            return {"test_cases": matches}
        
        return None