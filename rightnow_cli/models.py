from transformers import AutoModel, AutoConfig, PreTrainedModel
import torch
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from rich.console import Console
import json

console = Console()


class ModelLoader:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a HuggingFace model and extract relevant information."""
        console.print(f"[cyan]Loading model: {model_name}[/cyan]")
        
        try:
            config = AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            model_info = self._extract_model_info(config, model_name)
            
            model_info["supported_operations"] = self._identify_operations(config)
            
            console.print(f"[green]Model loaded successfully![/green]")
            console.print(f"  Model type: {model_info['model_type']}")
            console.print(f"  Hidden size: {model_info['hidden_size']}")
            console.print(f"  Number of layers: {model_info['num_layers']}")
            console.print(f"  Supported operations: {', '.join(model_info['supported_operations'])}")
            
            return model_info
            
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            console.print("[yellow]Using default model configuration[/yellow]")
            return self._get_default_model_info(model_name)
    
    def _extract_model_info(self, config: Any, model_name: str) -> Dict[str, Any]:
        """Extract relevant information from model config."""
        info = {
            "name": model_name,
            "model_type": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", 4096),
            "num_layers": getattr(config, "num_hidden_layers", 32),
            "num_attention_heads": getattr(config, "num_attention_heads", 32),
            "intermediate_size": getattr(config, "intermediate_size", 11008),
            "max_seq_len": getattr(config, "max_position_embeddings", 2048),
            "vocab_size": getattr(config, "vocab_size", 32000),
            "batch_size": 1,
            "dtype": "float32"
        }
        
        if hasattr(config, "num_key_value_heads"):
            info["num_key_value_heads"] = config.num_key_value_heads
        else:
            info["num_key_value_heads"] = info["num_attention_heads"]
        
        if hasattr(config, "hidden_act"):
            info["activation"] = config.hidden_act
        elif hasattr(config, "activation_function"):
            info["activation"] = config.activation_function
        else:
            info["activation"] = "gelu"
        
        if hasattr(config, "layer_norm_eps"):
            info["layer_norm_eps"] = config.layer_norm_eps
        elif hasattr(config, "layer_norm_epsilon"):
            info["layer_norm_eps"] = config.layer_norm_epsilon
        else:
            info["layer_norm_eps"] = 1e-5
        
        if hasattr(config, "rope_theta"):
            info["rope_theta"] = config.rope_theta
        elif hasattr(config, "rotary_emb_base"):
            info["rope_theta"] = config.rotary_emb_base
        else:
            info["rope_theta"] = 10000.0
        
        return info
    
    def _identify_operations(self, config: Any) -> List[str]:
        """Identify which operations are used in the model."""
        operations = ["matmul"]
        
        model_type = getattr(config, "model_type", "").lower()
        
        if model_type in ["llama", "mistral", "gpt2", "gpt_neo", "gpt_neox", "opt", "bloom"]:
            operations.extend(["attention", "layernorm"])
            
            activation = getattr(config, "hidden_act", "gelu")
            if "gelu" in activation.lower():
                operations.append("gelu")
            elif "silu" in activation.lower() or "swish" in activation.lower():
                operations.append("silu")
            elif "relu" in activation.lower():
                operations.append("relu")
        
        elif model_type in ["bert", "roberta", "electra"]:
            operations.extend(["attention", "layernorm", "gelu"])
        
        elif model_type in ["t5", "bart"]:
            operations.extend(["attention", "layernorm", "relu"])
        
        else:
            operations.extend(["attention", "layernorm", "gelu"])
        
        operations.append("softmax")
        
        return list(set(operations))
    
    def _get_default_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get default model info when loading fails."""
        common_configs = {
            "mistral-7b": {
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "max_seq_len": 32768,
                "vocab_size": 32000,
                "activation": "silu",
                "model_type": "mistral"
            },
            "llama-7b": {
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "intermediate_size": 11008,
                "max_seq_len": 2048,
                "vocab_size": 32000,
                "activation": "silu",
                "model_type": "llama"
            },
            "gpt2": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "intermediate_size": 3072,
                "max_seq_len": 1024,
                "vocab_size": 50257,
                "activation": "gelu",
                "model_type": "gpt2"
            }
        }
        
        base_name = model_name.lower()
        for key, config in common_configs.items():
            if key in base_name:
                info = {
                    "name": model_name,
                    "batch_size": 1,
                    "dtype": "float32",
                    "layer_norm_eps": 1e-5,
                    "rope_theta": 10000.0,
                    **config
                }
                info["supported_operations"] = ["matmul", "attention", "layernorm", 
                                               config["activation"], "softmax"]
                return info
        
        return {
            "name": model_name,
            "model_type": "unknown",
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 2048,
            "vocab_size": 32000,
            "batch_size": 1,
            "dtype": "float32",
            "activation": "gelu",
            "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "supported_operations": ["matmul", "attention", "layernorm", "gelu", "softmax"]
        }
    
    def estimate_memory_requirements(self, model_info: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory requirements for the model."""
        hidden_size = model_info["hidden_size"]
        num_layers = model_info["num_layers"]
        vocab_size = model_info["vocab_size"]
        intermediate_size = model_info["intermediate_size"]
        max_seq_len = model_info["max_seq_len"]
        batch_size = model_info["batch_size"]
        bytes_per_param = 4
        
        embedding_params = vocab_size * hidden_size
        
        attention_params_per_layer = (
            4 * hidden_size * hidden_size
        )
        
        mlp_params_per_layer = (
            2 * hidden_size * intermediate_size + intermediate_size
        )
        
        norm_params_per_layer = 2 * hidden_size
        
        total_params_per_layer = (
            attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer
        )
        
        total_params = embedding_params + (num_layers * total_params_per_layer) + hidden_size
        
        model_size_gb = (total_params * bytes_per_param) / (1024**3)
        
        kv_cache_per_layer = 2 * batch_size * max_seq_len * hidden_size * bytes_per_param
        total_kv_cache_gb = (kv_cache_per_layer * num_layers) / (1024**3)
        
        activations_per_layer = batch_size * max_seq_len * (
            hidden_size * 4 +
            intermediate_size * 2
        ) * bytes_per_param
        
        peak_activations_gb = (activations_per_layer * 2) / (1024**3)
        
        return {
            "model_parameters": total_params,
            "model_size_gb": model_size_gb,
            "kv_cache_gb": total_kv_cache_gb,
            "peak_activations_gb": peak_activations_gb,
            "total_memory_gb": model_size_gb + total_kv_cache_gb + peak_activations_gb
        }
    
    def get_operation_dimensions(
        self,
        model_info: Dict[str, Any],
        operation: str
    ) -> List[Tuple[str, Dict[str, int]]]:
        """Get typical dimensions for different operations in the model."""
        hidden_size = model_info["hidden_size"]
        num_heads = model_info["num_attention_heads"]
        head_dim = hidden_size // num_heads
        intermediate_size = model_info["intermediate_size"]
        seq_len = min(model_info["max_seq_len"], 512)
        batch_size = model_info["batch_size"]
        
        if operation == "matmul":
            return [
                ("QKV projection", {
                    "M": batch_size * seq_len,
                    "K": hidden_size,
                    "N": hidden_size * 3
                }),
                ("Output projection", {
                    "M": batch_size * seq_len,
                    "K": hidden_size,
                    "N": hidden_size
                }),
                ("FFN up projection", {
                    "M": batch_size * seq_len,
                    "K": hidden_size,
                    "N": intermediate_size
                }),
                ("FFN down projection", {
                    "M": batch_size * seq_len,
                    "K": intermediate_size,
                    "N": hidden_size
                })
            ]
        
        elif operation == "attention":
            return [
                ("Self-attention", {
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "seq_len": seq_len,
                    "head_dim": head_dim
                })
            ]
        
        elif operation == "layernorm":
            return [
                ("Layer normalization", {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_size": hidden_size
                })
            ]
        
        elif operation in ["gelu", "silu", "relu"]:
            return [
                (f"{operation.upper()} activation", {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_size": intermediate_size
                })
            ]
        
        elif operation == "softmax":
            return [
                ("Attention softmax", {
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "seq_len_q": seq_len,
                    "seq_len_k": seq_len
                })
            ]
        
        else:
            return []
    
    def save_model_info(self, model_info: Dict[str, Any], output_path: Path):
        """Save model information to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        console.print(f"[green]Model info saved to {output_path}[/green]")
    
    def load_model_info(self, input_path: Path) -> Dict[str, Any]:
        """Load model information from a JSON file."""
        with open(input_path, 'r') as f:
            return json.load(f)