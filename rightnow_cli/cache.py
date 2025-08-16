import json
import os
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import pickle

class CacheManager:
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            self.base_dir = Path.home() / ".ultrathink-cli"
        else:
            self.base_dir = Path(base_dir)
        
        self.config_dir = self.base_dir
        self.cache_dir = self.base_dir / "cache"
        self.kernels_dir = self.cache_dir / "kernels"
        self.db_path = self.cache_dir / "metadata.db"
        
        self._initialize_directories()
        self._initialize_database()
    
    def _initialize_directories(self):
        """Create necessary directories if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kernels_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self):
        """Initialize SQLite database for metadata storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kernel_metadata (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_used TIMESTAMP NOT NULL,
                    performance_metrics TEXT,
                    constraints TEXT,
                    file_path TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    baseline_time_ms REAL,
                    optimized_time_ms REAL,
                    speedup REAL,
                    memory_usage_mb REAL,
                    notes TEXT
                )
            """)
            conn.commit()
    
    def get_config_path(self) -> Path:
        """Get the config file path."""
        return self.config_dir / "config.json"
    
    def get_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = self.get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        config_path = self.get_config_path()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def has_api_key(self) -> bool:
        """Check if API key is configured."""
        config = self.get_config()
        return bool(config.get('openrouter_api_key'))
    
    def get_api_key(self) -> Optional[str]:
        """Get the stored API key."""
        config = self.get_config()
        return config.get('openrouter_api_key')
    
    def save_api_key(self, api_key: str):
        """Save API key to config."""
        config = self.get_config()
        config['openrouter_api_key'] = api_key
        self.save_config(config)
    
    def _generate_kernel_id(self, model_name: str, operation: str, code: str) -> str:
        """Generate unique ID for a kernel."""
        content = f"{model_name}:{operation}:{code}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def cache_kernel(self, model_name: str, operation: str, kernel_data: Dict[str, Any]):
        """Cache a kernel and its metadata."""
        kernel_code = kernel_data.get("code", "")
        kernel_id = self._generate_kernel_id(model_name, operation, kernel_code)
        
        kernel_file = self.kernels_dir / f"{kernel_id}.pkl"
        with open(kernel_file, 'wb') as f:
            pickle.dump(kernel_data, f)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO kernel_metadata 
                (id, model_name, operation, created_at, last_used, 
                 performance_metrics, constraints, file_path, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kernel_id,
                model_name,
                operation,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps(kernel_data.get("benchmark", {})),
                json.dumps(kernel_data.get("constraints", {})),
                str(kernel_file),
                hashlib.sha256(kernel_code.encode()).hexdigest()
            ))
            conn.commit()
        
        return kernel_id
    
    def get_cached_kernel(self, model_name: str, operation: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached kernel if available."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, file_path FROM kernel_metadata
                WHERE model_name = ? AND operation = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (model_name, operation))
            
            result = cursor.fetchone()
            if result:
                kernel_id, file_path = result
                kernel_file = Path(file_path)
                
                if kernel_file.exists():
                    cursor.execute("""
                        UPDATE kernel_metadata
                        SET last_used = ?
                        WHERE id = ?
                    """, (datetime.now().isoformat(), kernel_id))
                    conn.commit()
                    
                    with open(kernel_file, 'rb') as f:
                        return pickle.load(f)
        
        return None
    
    def list_cached_kernels(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all cached kernels, optionally filtered by model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_name:
                query = """
                    SELECT model_name, operation, created_at, performance_metrics, file_path
                    FROM kernel_metadata
                    WHERE model_name = ?
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (model_name,))
            else:
                query = """
                    SELECT model_name, operation, created_at, performance_metrics, file_path
                    FROM kernel_metadata
                    ORDER BY created_at DESC
                """
                cursor.execute(query)
            
            results = []
            for row in cursor.fetchall():
                model, op, created, perf_json, file_path = row
                file_size = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
                
                results.append({
                    "model": model,
                    "operation": op,
                    "created": created,
                    "performance": json.loads(perf_json) if perf_json else {},
                    "size_kb": file_size
                })
            
            return results
    
    def count_cached_kernels(self) -> int:
        """Count total number of cached kernels."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM kernel_metadata")
            return cursor.fetchone()[0]
    
    def clear_cache(self):
        """Clear all cached kernels."""
        shutil.rmtree(self.kernels_dir)
        self.kernels_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM kernel_metadata")
            conn.commit()
    
    def record_optimization(
        self,
        model_name: str,
        operation: str,
        baseline_time_ms: float,
        optimized_time_ms: float,
        memory_usage_mb: float,
        notes: Optional[str] = None
    ):
        """Record optimization results in history."""
        speedup = baseline_time_ms / optimized_time_ms if optimized_time_ms > 0 else 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO optimization_history
                (model_name, operation, timestamp, baseline_time_ms, 
                 optimized_time_ms, speedup, memory_usage_mb, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                operation,
                datetime.now().isoformat(),
                baseline_time_ms,
                optimized_time_ms,
                speedup,
                memory_usage_mb,
                notes
            ))
            conn.commit()
    
    def get_optimization_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get optimization history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_name:
                query = """
                    SELECT * FROM optimization_history
                    WHERE model_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor.execute(query, (model_name, limit))
            else:
                query = """
                    SELECT * FROM optimization_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]