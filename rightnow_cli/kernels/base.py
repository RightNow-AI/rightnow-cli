from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseKernel(ABC):
    """Base class for CUDA kernel implementations."""
    
    def __init__(self, operation: str):
        self.operation = operation
    
    @abstractmethod
    def get_kernel_code(self) -> str:
        """Return the CUDA kernel code."""
        pass
    
    @abstractmethod
    def get_launch_config(self, problem_size: Dict[str, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Return the launch configuration (grid_size, block_size) for the kernel.
        
        Args:
            problem_size: Dictionary containing problem dimensions
            
        Returns:
            Tuple of (grid_size, block_size) where each is a 3-tuple
        """
        pass
    
    @abstractmethod
    def get_kernel_name(self) -> str:
        """Return the kernel function name."""
        pass
    
    def get_constraints(self) -> Dict[str, Any]:
        """Return kernel constraints (registers, shared memory, etc)."""
        return {
            "max_registers": 255,
            "shared_memory_kb": 48,
            "max_threads_per_block": 1024
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return kernel metadata."""
        return {
            "operation": self.operation,
            "kernel_name": self.get_kernel_name(),
            "constraints": self.get_constraints()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert kernel to dictionary format."""
        return {
            "code": self.get_kernel_code(),
            "operation": self.operation,
            "kernel_name": self.get_kernel_name(),
            "constraints": self.get_constraints(),
            "metadata": self.get_metadata()
        }