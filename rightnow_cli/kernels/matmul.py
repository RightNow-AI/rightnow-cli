from .base import BaseKernel
from typing import Dict, Tuple


class MatMulKernel(BaseKernel):
    """Baseline matrix multiplication kernel implementation."""
    
    def __init__(self):
        super().__init__("matmul")
    
    def get_kernel_code(self) -> str:
        """Return the CUDA kernel code for matrix multiplication."""
        return """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int K,
    const int N
) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    
    float Cvalue = 0.0f;
    
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A
        if (Row < M && t * TILE_WIDTH + tx < K) {
            As[ty][tx] = A[Row * K + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        if (t * TILE_WIDTH + ty < K && Col < N) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + Col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

// Optimized version with better memory access patterns
__global__ void matmul_kernel_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int K,
    const int N
) {
    const int TILE_WIDTH_A = 64;
    const int TILE_WIDTH_B = 16;
    
    __shared__ float As[TILE_WIDTH_A][TILE_WIDTH_B + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_WIDTH_B][TILE_WIDTH_A + 1];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int aRow = blockIdx.y * TILE_WIDTH_A + threadIdx.y;
    int bCol = blockIdx.x * TILE_WIDTH_A + threadIdx.x;
    
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int tileK = 0; tileK < K; tileK += TILE_WIDTH_B) {
        // Collaborative loading of tiles
        for (int i = tid; i < TILE_WIDTH_A * TILE_WIDTH_B; i += blockDim.x * blockDim.y) {
            int row = i / TILE_WIDTH_B;
            int col = i % TILE_WIDTH_B;
            int aIdx = (blockIdx.y * TILE_WIDTH_A + row) * K + tileK + col;
            if (blockIdx.y * TILE_WIDTH_A + row < M && tileK + col < K) {
                As[row][col] = A[aIdx];
            } else {
                As[row][col] = 0.0f;
            }
        }
        
        for (int i = tid; i < TILE_WIDTH_B * TILE_WIDTH_A; i += blockDim.x * blockDim.y) {
            int row = i / TILE_WIDTH_A;
            int col = i % TILE_WIDTH_A;
            int bIdx = (tileK + row) * N + blockIdx.x * TILE_WIDTH_A + col;
            if (tileK + row < K && blockIdx.x * TILE_WIDTH_A + col < N) {
                Bs[row][col] = B[bIdx];
            } else {
                Bs[row][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH_B; ++k) {
            float aVal = As[threadIdx.y][k];
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                if (threadIdx.x + n * 16 < TILE_WIDTH_A) {
                    sum[n] += aVal * Bs[k][threadIdx.x + n * 16];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int col = bCol + n * 16;
        if (aRow < M && col < N) {
            C[aRow * N + col] = sum[n];
        }
    }
}

// Wrapper kernel that selects the best implementation
extern "C" __global__ void matmul_kernel_dispatch(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int K,
    const int N
) {
    // For small matrices, use the simple tiled version
    if (M * N < 65536) {
        matmul_kernel<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(
            A, B, C, M, K, N
        );
    } else {
        // For larger matrices, use the optimized version
        matmul_kernel_v2<<<dim3((N + 63) / 64, (M + 63) / 64), dim3(16, 4)>>>(
            A, B, C, M, K, N
        );
    }
}
"""
    
    def get_launch_config(self, problem_size: Dict[str, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return the launch configuration for the kernel."""
        M = problem_size.get("M", 1024)
        N = problem_size.get("N", 1024)
        
        # Use simple tiled version for small matrices
        if M * N < 65536:
            block_size = (16, 16, 1)
            grid_size = ((N + 15) // 16, (M + 15) // 16, 1)
        else:
            # Use optimized version for larger matrices
            block_size = (16, 4, 1)
            grid_size = ((N + 63) // 64, (M + 63) // 64, 1)
        
        return grid_size, block_size
    
    def get_kernel_name(self) -> str:
        """Return the kernel function name."""
        return "matmul_kernel"
    
    def get_constraints(self) -> Dict[str, Any]:
        """Return kernel constraints."""
        return {
            "max_registers": 64,
            "shared_memory_kb": 48,
            "max_threads_per_block": 256
        }