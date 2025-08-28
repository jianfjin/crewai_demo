"""
RAG Document Monitor with Automatic File Monitoring and Dynamic Document Discovery
"""

import os
import glob
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import watchdog for file system monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None


class DocumentChangeHandler(FileSystemEventHandler):
    """Handle file system events for document changes."""
    
    def __init__(self, callback: Callable[[str, str], None]):
        """
        Initialize the handler.
        
        Args:
            callback: Function to call when files change (file_path, event_type)
        """
        self.callback = callback
        self.supported_extensions = {'.md', '.txt', '.py', '.yaml', '.yml', '.json'}
        
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            file_path = event.src_path
            if self._is_supported_file(file_path):
                logger.info(f"📝 File modified: {file_path}")
                self.callback(file_path, "modified")
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            file_path = event.src_path
            if self._is_supported_file(file_path):
                logger.info(f"📄 New file created: {file_path}")
                self.callback(file_path, "created")
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            file_path = event.src_path
            if self._is_supported_file(file_path):
                logger.info(f"🗑️ File deleted: {file_path}")
                self.callback(file_path, "deleted")
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported for indexing."""
        return Path(file_path).suffix.lower() in self.supported_extensions


class RAGDocumentMonitor:
    """
    Monitor and manage RAG knowledge base documents with automatic updates.
    """
    
    def __init__(self, knowledge_base=None):
        """
        Initialize the document monitor.
        
        Args:
            knowledge_base: RAG knowledge base instance
        """
        self.knowledge_base = knowledge_base
        self.observer = None
        self.monitoring = False
        self.stop_monitoring_flag = False
        self.watched_directories = set()
        self.update_queue = []
        self.update_thread = None
        
        # Document discovery patterns
        self.discovery_patterns = [
            "**/*.md",
            "**/*.txt", 
            "**/*.py",
            "**/*.yaml",
            "**/*.yml",
            "**/*.json"
        ]
        
        # Exclude patterns
        self.exclude_patterns = [
            "**/node_modules/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.env",
            "**/tmp_*",
            "**/*.pyc",
            "**/*.log"
        ]
        
    def start_monitoring(self, directories: List[str] = None) -> bool:
        """
        Start monitoring directories for file changes.
        
        Args:
            directories: List of directories to monitor. If None, uses default directories.
            
        Returns:
            bool: True if monitoring started successfully
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("⚠️ Watchdog not available. File monitoring disabled.")
            return False
            
        if self.monitoring:
            logger.info("📁 File monitoring already active")
            return True
            
        try:
            # Reset stop flag
            self.stop_monitoring_flag = False
            
            # Use default directories if none provided
            if directories is None:
                directories = self._get_default_directories()
            
            # Initialize observer
            self.observer = Observer()
            handler = DocumentChangeHandler(self._handle_file_change)
            
            # Add directories to watch
            for directory in directories:
                if os.path.exists(directory):
                    self.observer.schedule(handler, directory, recursive=True)
                    self.watched_directories.add(directory)
                    logger.info(f"👁️ Watching directory: {directory}")
                else:
                    logger.warning(f"⚠️ Directory not found: {directory}")
            
            # Start observer
            self.observer.start()
            self.monitoring = True
            
            # Start update processing thread
            self._start_update_thread()
            
            logger.info("🚀 RAG document monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop file monitoring."""
        if not self.monitoring:
            return
            
        try:
            self.stop_monitoring_flag = True
            
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            self.monitoring = False
            self.watched_directories.clear()
            logger.info("🛑 RAG document monitoring stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring: {e}")
    
    def discover_documents(self, base_directory: str = ".") -> List[Dict[str, Any]]:
        """
        Discover all documents in the specified directory.
        
        Args:
            base_directory: Base directory to search
            
        Returns:
            List of discovered documents with metadata
        """
        discovered_docs = []
        base_path = Path(base_directory).resolve()
        
        try:
            for pattern in self.discovery_patterns:
                for file_path in base_path.glob(pattern):
                    if self._should_include_file(file_path):
                        doc_info = self._get_document_info(file_path)
                        discovered_docs.append(doc_info)
            
            # Sort by modification time (newest first)
            discovered_docs.sort(key=lambda x: x['modified_time'], reverse=True)
            
            logger.info(f"🔍 Discovered {len(discovered_docs)} documents in {base_directory}")
            return discovered_docs
            
        except Exception as e:
            logger.error(f"❌ Error discovering documents: {e}")
            return []
    
    def add_document_to_rag(self, file_path: str, force_reindex: bool = False) -> bool:
        """
        Add a document to the RAG knowledge base.
        
        Args:
            file_path: Path to the document
            force_reindex: Force reindexing even if file hasn't changed
            
        Returns:
            bool: True if document was added successfully
        """
        if not self.knowledge_base:
            logger.warning("⚠️ No knowledge base available")
            return False
            
        try:
            success = self.knowledge_base.index_document(file_path, force_reindex=force_reindex)
            if success:
                logger.info(f"✅ Added document to RAG: {file_path}")
            else:
                logger.warning(f"⚠️ Failed to add document: {file_path}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error adding document to RAG: {e}")
            return False
    
    def remove_document_from_rag(self, file_path: str) -> bool:
        """
        Remove a document from the RAG knowledge base.
        
        Args:
            file_path: Path to the document to remove
            
        Returns:
            bool: True if document was removed successfully
        """
        if not self.knowledge_base:
            logger.warning("⚠️ No knowledge base available")
            return False
            
        try:
            # Remove from indexed documents
            if hasattr(self.knowledge_base, 'indexed_documents'):
                if file_path in self.knowledge_base.indexed_documents:
                    del self.knowledge_base.indexed_documents[file_path]
                    logger.info(f"🗑️ Removed document from RAG: {file_path}")
                    return True
            
            logger.warning(f"⚠️ Document not found in RAG: {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error removing document from RAG: {e}")
            return False
    
    def update_rag_knowledge_base(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Update the entire RAG knowledge base.
        
        Args:
            force_rebuild: Force rebuilding all documents
            
        Returns:
            Dictionary with update results
        """
        if not self.knowledge_base:
            return {"error": "No knowledge base available"}
            
        try:
            logger.info("🔄 Updating RAG knowledge base...")
            result = self.knowledge_base.build_knowledge_base(force_rebuild=force_rebuild)
            logger.info(f"✅ RAG knowledge base updated: {result.get('files_indexed', 0)} files")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error updating RAG knowledge base: {e}")
            return {"error": str(e)}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "monitoring": self.monitoring,
            "watchdog_available": WATCHDOG_AVAILABLE,
            "watched_directories": list(self.watched_directories),
            "pending_updates": len(self.update_queue),
            "knowledge_base_available": self.knowledge_base is not None
        }
    
    def _handle_file_change(self, file_path: str, event_type: str):
        """Handle file change events."""
        # Add to update queue
        self.update_queue.append({
            "file_path": file_path,
            "event_type": event_type,
            "timestamp": datetime.now()
        })
    
    def _start_update_thread(self):
        """Start the update processing thread."""
        def process_updates():
            while not getattr(self, 'stop_monitoring_flag', False):
                try:
                    if self.update_queue:
                        # Process pending updates
                        updates_to_process = self.update_queue.copy()
                        self.update_queue.clear()
                        
                        for update in updates_to_process:
                            file_path = update["file_path"]
                            event_type = update["event_type"]
                            
                            if event_type in ["created", "modified"]:
                                self.add_document_to_rag(file_path, force_reindex=True)
                            elif event_type == "deleted":
                                self.remove_document_from_rag(file_path)
                    
                    time.sleep(2)  # Check every 2 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error processing updates: {e}")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(target=process_updates, daemon=True)
        self.update_thread.start()
    
    def _get_default_directories(self) -> List[str]:
        """Get default directories to monitor."""
        return [
            ".",  # Current directory
            "src",
            "docs",
            "config"
        ]
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included based on exclude patterns."""
        file_str = str(file_path)
        
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return False
        
        return True
    
    def _get_document_info(self, file_path: Path) -> Dict[str, Any]:
        """Get document information."""
        try:
            stat = file_path.stat()
            return {
                "path": str(file_path),
                "name": file_path.name,
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime),
                "extension": file_path.suffix,
                "relative_path": str(file_path.relative_to(Path.cwd()))
            }
        except Exception as e:
            logger.error(f"❌ Error getting document info for {file_path}: {e}")
            return {
                "path": str(file_path),
                "name": file_path.name,
                "error": str(e)
            }