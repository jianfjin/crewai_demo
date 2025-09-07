"""
Persistent Context Storage with LangGraph Checkpointing
Implements Feature 1: Scratchpad Implementation with persistent context storage
"""

import os
import json
import sqlite3
import pickle
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import threading
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ContextCheckpoint:
    """Represents a context checkpoint with metadata"""
    checkpoint_id: str
    workflow_id: str
    agent_id: str
    timestamp: datetime
    context_data: Dict[str, Any]
    token_count: int
    priority_level: str
    dependencies: List[str]
    compression_level: str
    metadata: Dict[str, Any]


class PersistentContextStorage:
    """
    Persistent context storage system with LangGraph-style checkpointing.
    Provides durable storage for context data across workflow executions.
    """
    
    def __init__(self, storage_path: str = "cache/context_storage.db"):
        """Initialize persistent storage with SQLite backend."""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context_data BLOB NOT NULL,
                    token_count INTEGER NOT NULL,
                    priority_level TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    compression_level TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_agent 
                ON context_checkpoints(workflow_id, agent_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON context_checkpoints(timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper locking."""
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def save_checkpoint(
        self,
        workflow_id: str,
        agent_id: str,
        context_data: Dict[str, Any],
        token_count: int,
        priority_level: str = "normal",
        dependencies: List[str] = None,
        compression_level: str = "none",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Save a context checkpoint to persistent storage."""
        
        checkpoint_id = f"{workflow_id}_{agent_id}_{int(datetime.now().timestamp())}"
        dependencies = dependencies or []
        metadata = metadata or {}
        
        checkpoint = ContextCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            timestamp=datetime.now(),
            context_data=context_data,
            token_count=token_count,
            priority_level=priority_level,
            dependencies=dependencies,
            compression_level=compression_level,
            metadata=metadata
        )
        
        try:
            with self._get_connection() as conn:
                # Serialize complex data with enum handling
                context_blob = pickle.dumps(self._serialize_for_storage(context_data))
                dependencies_json = json.dumps(dependencies)
                metadata_json = json.dumps(self._serialize_for_storage(metadata))
                now = datetime.now().isoformat()
                
                conn.execute("""
                    INSERT OR REPLACE INTO context_checkpoints 
                    (checkpoint_id, workflow_id, agent_id, timestamp, context_data,
                     token_count, priority_level, dependencies, compression_level,
                     metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_id, workflow_id, agent_id, checkpoint.timestamp.isoformat(),
                    context_blob, token_count, priority_level, dependencies_json,
                    compression_level, metadata_json, now, now
                ))
                
                logger.info(f"Saved checkpoint {checkpoint_id} for {workflow_id}/{agent_id}")
                return checkpoint_id
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def _serialize_for_storage(self, data: Any) -> Any:
        """Recursively serialize data, converting enums to their values."""
        if isinstance(data, Enum):
            return data.value
        elif isinstance(data, dict):
            return {key: self._serialize_for_storage(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_storage(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._serialize_for_storage(item) for item in data)
        else:
            return data
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ContextCheckpoint]:
        """Load a specific checkpoint from storage."""
        
        try:
            with self._get_connection() as conn:
                row = conn.execute("""
                    SELECT * FROM context_checkpoints WHERE checkpoint_id = ?
                """, (checkpoint_id,)).fetchone()
                
                if not row:
                    return None
                
                # Deserialize data
                context_data = pickle.loads(row['context_data'])
                dependencies = json.loads(row['dependencies'])
                metadata = json.loads(row['metadata'])
                
                return ContextCheckpoint(
                    checkpoint_id=row['checkpoint_id'],
                    workflow_id=row['workflow_id'],
                    agent_id=row['agent_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    context_data=context_data,
                    token_count=row['token_count'],
                    priority_level=row['priority_level'],
                    dependencies=dependencies,
                    compression_level=row['compression_level'],
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_workflow_checkpoints(
        self,
        workflow_id: str,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ContextCheckpoint]:
        """Get all checkpoints for a workflow, optionally filtered by agent."""
        
        try:
            with self._get_connection() as conn:
                if agent_id:
                    query = """
                        SELECT * FROM context_checkpoints 
                        WHERE workflow_id = ? AND agent_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """
                    params = (workflow_id, agent_id, limit)
                else:
                    query = """
                        SELECT * FROM context_checkpoints 
                        WHERE workflow_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """
                    params = (workflow_id, limit)
                
                rows = conn.execute(query, params).fetchall()
                
                checkpoints = []
                for row in rows:
                    context_data = pickle.loads(row['context_data'])
                    dependencies = json.loads(row['dependencies'])
                    metadata = json.loads(row['metadata'])
                    
                    checkpoints.append(ContextCheckpoint(
                        checkpoint_id=row['checkpoint_id'],
                        workflow_id=row['workflow_id'],
                        agent_id=row['agent_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        context_data=context_data,
                        token_count=row['token_count'],
                        priority_level=row['priority_level'],
                        dependencies=dependencies,
                        compression_level=row['compression_level'],
                        metadata=metadata
                    ))
                
                return checkpoints
                
        except Exception as e:
            logger.error(f"Failed to get workflow checkpoints: {e}")
            return []
    
    def restore_workflow_context(
        self,
        workflow_id: str,
        target_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Restore complete workflow context from checkpoints."""
        
        checkpoints = self.get_workflow_checkpoints(workflow_id)
        
        if target_timestamp:
            # Filter checkpoints up to target timestamp
            checkpoints = [
                cp for cp in checkpoints 
                if cp.timestamp <= target_timestamp
            ]
        
        # Merge context data from all agents
        merged_context = {}
        agent_contexts = {}
        
        for checkpoint in checkpoints:
            agent_contexts[checkpoint.agent_id] = checkpoint.context_data
            merged_context.update(checkpoint.context_data)
        
        return {
            "merged_context": merged_context,
            "agent_contexts": agent_contexts,
            "checkpoint_count": len(checkpoints),
            "latest_timestamp": max(cp.timestamp for cp in checkpoints) if checkpoints else None
        }
    
    def cleanup_old_checkpoints(self, retention_days: int = 30):
        """Clean up checkpoints older than retention period."""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            with self._get_connection() as conn:
                result = conn.execute("""
                    DELETE FROM context_checkpoints 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old checkpoints")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage system."""
        
        try:
            with self._get_connection() as conn:
                # Count total checkpoints
                total_checkpoints = conn.execute(
                    "SELECT COUNT(*) as count FROM context_checkpoints"
                ).fetchone()['count']
                
                # Count by workflow
                workflow_counts = conn.execute("""
                    SELECT workflow_id, COUNT(*) as count 
                    FROM context_checkpoints 
                    GROUP BY workflow_id
                """).fetchall()
                
                # Count by agent
                agent_counts = conn.execute("""
                    SELECT agent_id, COUNT(*) as count 
                    FROM context_checkpoints 
                    GROUP BY agent_id
                """).fetchall()
                
                # Total token usage
                total_tokens = conn.execute(
                    "SELECT SUM(token_count) as total FROM context_checkpoints"
                ).fetchone()['total'] or 0
                
                # Storage size
                storage_size = os.path.getsize(self.storage_path) if self.storage_path.exists() else 0
                
                return {
                    "total_checkpoints": total_checkpoints,
                    "total_tokens": total_tokens,
                    "storage_size_mb": storage_size / (1024 * 1024),
                    "workflow_counts": {row['workflow_id']: row['count'] for row in workflow_counts},
                    "agent_counts": {row['agent_id']: row['count'] for row in agent_counts}
                }
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}


# Global instance
_global_persistent_storage = None


def get_persistent_storage() -> PersistentContextStorage:
    """Get the global persistent storage instance."""
    global _global_persistent_storage
    if _global_persistent_storage is None:
        _global_persistent_storage = PersistentContextStorage()
    return _global_persistent_storage