import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import backoff
from rich.console import Console

console = Console()


@dataclass
class KernelCandidate:
    code: str
    operation: str
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]


class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ultrathink-cli",
            "X-Title": "ultrathink-cli"
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=3,
        max_time=60
    )
    def _make_request(self, prompt: str, system_prompt: str, model: str = "openai/gpt-4") -> str:
        """Make a request to OpenRouter API with retry logic."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        response = requests.post(
            self.BASE_URL,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 401:
            raise ValueError("Invalid API key. Please check your OpenRouter API key.")
        
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_kernel_optimizations(
        self,
        original_code: str,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any],
        num_variants: int = 3
    ) -> List[KernelCandidate]:
        """Generate optimized variants of a user-provided CUDA kernel."""
        
        system_prompt = self._create_kernel_optimization_prompt()
        
        user_prompt = f"""Optimize this CUDA kernel for better performance:

```cuda
{original_code}
```

Kernel Analysis:
- Name: {analysis.get('kernel_name', 'unknown')}
- Current patterns: {', '.join(analysis.get('patterns', []))}
- Arithmetic intensity: {analysis.get('arithmetic_intensity', 0):.2f}
- Complexity: {analysis.get('complexity', 'unknown')}

Constraints:
- Max registers: {constraints.get('max_registers', 255)}
- Max shared memory: {constraints.get('shared_memory_kb', 48)}KB
- Target GPU: {constraints.get('target_gpu', 'sm_70')}

Please generate {num_variants} optimized variants with different optimization strategies.
Focus on: memory coalescing, shared memory usage, instruction-level parallelism, and minimizing divergence."""
        
        console.print(f"[cyan]Generating {num_variants} optimization variants...[/cyan]")
        
        candidates = []
        for i in range(num_variants):
            try:
                response = self._make_request(user_prompt + f"\n\nGenerate variant {i+1} with a different optimization approach.", system_prompt)
                kernel_code = self._extract_cuda_code(response)
                
                if kernel_code:
                    candidate = KernelCandidate(
                        code=kernel_code,
                        operation=analysis.get('kernel_name', 'optimized_kernel'),
                        constraints=constraints,
                        metadata={
                            "variant": i + 1,
                            "original_analysis": analysis,
                            "generated_at": time.time()
                        }
                    )
                    candidates.append(candidate)
                    console.print(f"[green]Generated optimization variant {i + 1}/{num_variants}[/green]")
                else:
                    console.print(f"[yellow]Failed to extract valid CUDA code for variant {i + 1}[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]Error generating variant {i + 1}: {e}[/red]")
        
        return candidates
    
    def generate_kernels(
        self,
        operation: str,
        model_info: Dict[str, Any],
        constraints: Dict[str, Any],
        num_candidates: int = 3
    ) -> List[KernelCandidate]:
        """Generate CUDA kernel candidates for a specific operation."""
        
        system_prompt = self._create_system_prompt()
        
        user_prompt = self._create_kernel_generation_prompt(
            operation, model_info, constraints
        )
        
        console.print(f"[cyan]Generating {num_candidates} kernel candidates for {operation}...[/cyan]")
        
        candidates = []
        for i in range(num_candidates):
            try:
                response = self._make_request(user_prompt, system_prompt)
                kernel_code = self._extract_cuda_code(response)
                
                if kernel_code:
                    candidate = KernelCandidate(
                        code=kernel_code,
                        operation=operation,
                        constraints=constraints,
                        metadata={
                            "variant": i + 1,
                            "model_info": model_info,
                            "generated_at": time.time()
                        }
                    )
                    candidates.append(candidate)
                    console.print(f"[green]Generated kernel variant {i + 1}/{num_candidates}[/green]")
                else:
                    console.print(f"[yellow]Failed to extract valid CUDA code for variant {i + 1}[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]Error generating kernel variant {i + 1}: {e}[/red]")
        
        return candidates
    
    def _create_kernel_optimization_prompt(self) -> str:
        """Create the system prompt for general kernel optimization."""
        return """You are an expert CUDA performance engineer specializing in kernel optimization.

Your task is to optimize user-provided CUDA kernels by:
1. Analyzing the current implementation for bottlenecks
2. Applying advanced optimization techniques
3. Ensuring correctness is maintained
4. Maximizing performance for the target GPU

Key optimization strategies to consider:
- Memory coalescing and access patterns
- Shared memory utilization and bank conflict avoidance
- Warp-level primitives and shuffle operations
- Instruction-level parallelism and loop unrolling
- Register usage optimization
- Minimizing thread divergence
- Using vectorized loads/stores (float2, float4)
- Tensor Core utilization where applicable
- Optimal launch configuration

Always provide complete, compilable CUDA code with:
- Clear comments explaining optimizations
- Proper error checking
- Launch configuration recommendations
- Expected performance improvements"""
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for kernel generation."""
        return """You are an expert CUDA kernel developer specializing in high-performance GPU computing for AI workloads.

Your task is to generate optimized CUDA kernels that:
1. Are numerically correct and stable
2. Maximize GPU utilization (occupancy, memory bandwidth, compute throughput)
3. Follow CUDA best practices (coalesced memory access, shared memory usage, etc.)
4. Are compatible with PyTorch tensors
5. Include proper error checking and boundary conditions

When generating kernels, consider:
- Register usage and occupancy
- Shared memory optimization
- Warp-level primitives when beneficial
- Memory access patterns and bank conflicts
- Numerical precision requirements

Always provide complete, compilable CUDA C++ code with:
- Kernel function implementation
- Launch configuration recommendations
- Brief comments explaining optimization choices"""
    
    def _create_kernel_generation_prompt(
        self,
        operation: str,
        model_info: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Create the user prompt for kernel generation."""
        
        operation_prompts = {
            "matmul": self._matmul_prompt,
            "attention": self._attention_prompt,
            "layernorm": self._layernorm_prompt,
            "gelu": self._gelu_prompt,
            "softmax": self._softmax_prompt
        }
        
        if operation in operation_prompts:
            base_prompt = operation_prompts[operation](model_info)
        else:
            base_prompt = f"Generate an optimized CUDA kernel for {operation} operation."
        
        constraints_str = "\n".join([f"- {k}: {v}" for k, v in constraints.items()])
        
        return f"""{base_prompt}

Model Information:
- Model: {model_info.get('name', 'Unknown')}
- Hidden size: {model_info.get('hidden_size', 'Unknown')}
- Number of layers: {model_info.get('num_layers', 'Unknown')}
- Data type: {model_info.get('dtype', 'float32')}

Constraints:
{constraints_str}

Requirements:
1. The kernel must be numerically stable and correct
2. Include launch configuration (grid/block dimensions)
3. Optimize for the given constraints
4. Include basic error checking
5. Make the kernel compatible with PyTorch tensors

Please provide the complete CUDA kernel code."""
    
    def _matmul_prompt(self, model_info: Dict[str, Any]) -> str:
        """Create prompt for matrix multiplication kernel."""
        return f"""Generate an optimized CUDA kernel for matrix multiplication (GEMM).

The kernel should compute: C = alpha * A @ B + beta * C

Where:
- A is shape [M, K]
- B is shape [K, N]
- C is shape [M, N]
- Typical dimensions for this model: M={model_info.get('batch_size', 1)}, K={model_info.get('hidden_size', 4096)}, N={model_info.get('hidden_size', 4096)}

Consider using:
- Shared memory tiling
- Register blocking
- Vectorized loads/stores
- Tensor cores if available"""
    
    def _attention_prompt(self, model_info: Dict[str, Any]) -> str:
        """Create prompt for attention kernel."""
        head_dim = model_info.get('hidden_size', 4096) // model_info.get('num_attention_heads', 32)
        return f"""Generate an optimized CUDA kernel for scaled dot-product attention.

The kernel should compute attention scores for transformer models:
- Input: Q, K, V tensors of shape [batch, num_heads, seq_len, head_dim]
- Output: Attention output of same shape
- Head dimension: {head_dim}
- Number of heads: {model_info.get('num_attention_heads', 32)}

Consider:
- Flash Attention-style algorithm if beneficial
- Efficient softmax computation
- Memory-efficient backward pass support"""
    
    def _layernorm_prompt(self, model_info: Dict[str, Any]) -> str:
        """Create prompt for layer normalization kernel."""
        return f"""Generate an optimized CUDA kernel for layer normalization.

The kernel should:
- Normalize input tensor along the last dimension
- Apply learned scale (gamma) and shift (beta) parameters
- Hidden size: {model_info.get('hidden_size', 4096)}
- Support both forward and gradient computation

Consider:
- Welford's algorithm for numerical stability
- Vectorized operations
- Efficient use of shared memory for reductions"""
    
    def _gelu_prompt(self, model_info: Dict[str, Any]) -> str:
        """Create prompt for GELU activation kernel."""
        return """Generate an optimized CUDA kernel for GELU (Gaussian Error Linear Unit) activation.

The kernel should implement: GELU(x) = x * Phi(x)
Where Phi(x) is the cumulative distribution function of the standard normal distribution.

Consider:
- Fast approximations (tanh-based or polynomial)
- Vectorized operations
- Fused implementations if beneficial"""
    
    def _softmax_prompt(self, model_info: Dict[str, Any]) -> str:
        """Create prompt for softmax kernel."""
        return """Generate an optimized CUDA kernel for softmax operation.

The kernel should:
- Compute softmax along a specified dimension
- Handle numerical stability (subtract max before exp)
- Support variable sequence lengths

Consider:
- Efficient reduction patterns
- Warp-level primitives
- Online softmax algorithm for memory efficiency"""
    
    def _extract_cuda_code(self, response: str) -> Optional[str]:
        """Extract CUDA code from the API response."""
        import re
        
        cuda_pattern = r'```(?:cuda|cpp|c\+\+)?\n(.*?)```'
        matches = re.findall(cuda_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        if "__global__" in response and "void" in response:
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if "__global__" in line or "__device__" in line or "#include" in line:
                    in_code = True
                if in_code:
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines).strip()
        
        return None
    
    def validate_api_key(self) -> bool:
        """Validate the API key by making a test request."""
        try:
            test_prompt = "Hello, this is a test."
            test_system = "You are a helpful assistant."
            self._make_request(test_prompt, test_system)
            return True
        except ValueError as e:
            if "Invalid API key" in str(e):
                return False
            raise
        except Exception:
            return False