"""
RAG Knowledge Base for Marketing Research Tool

This module creates a comprehensive knowledge base from project documentation
and source code to enable intelligent chat assistance in the dashboard.

Features:
- Document ingestion and processing
- Vector embeddings for semantic search
- Agent and tool discovery
- Feature explanation and guidance
- Integration with existing Mem0 infrastructure
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import yaml
from pathlib import Path
import hashlib

# Vector store and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Text processing
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    source_file: str
    chunk_id: str
    metadata: Dict[str, Any]
    embedding_id: Optional[str] = None

class MarketingResearchKnowledgeBase:
    """
    Comprehensive knowledge base for the Marketing Research Tool.
    
    This class builds and maintains a searchable knowledge base from:
    - Documentation files (README, implementation guides, etc.)
    - Source code (agents, tools, workflows)
    - Configuration files (YAML configs)
    """
    
    def __init__(self, db_path: str = "./db/knowledge_base"):
        """
        Initialize the knowledge base.
        
        Args:
            db_path: Path to store the vector database
        """
        self.db_path = db_path
        self.collection_name = "marketing_research_kb"
        
        # Ensure db directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = None
        self.collection = None
        self._init_vector_store()
        
        # Document tracking
        self.indexed_documents = {}
        self.load_index_metadata()
        
        # Knowledge categories
        self.knowledge_categories = {
            "documentation": [],
            "agents": [],
            "tools": [],
            "workflows": [],
            "configurations": [],
            "implementations": []
        }
    
    def _init_vector_store(self):
        """Initialize the vector store (ChromaDB)."""
        try:
            if not CHROMA_AVAILABLE:
                logger.warning("ChromaDB not available, using fallback storage")
                return
            
            # Initialize ChromaDB client
            self.vector_store = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.vector_store.get_collection(self.collection_name)
                logger.info(f"✅ Loaded existing knowledge base collection: {self.collection_name}")
            except:
                self.collection = self.vector_store.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Marketing Research Tool Knowledge Base"}
                )
                logger.info(f"✅ Created new knowledge base collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            self.vector_store = None
            self.collection = None
    
    def load_index_metadata(self):
        """Load metadata about indexed documents."""
        metadata_file = os.path.join(self.db_path, "index_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.indexed_documents = json.load(f)
                logger.info(f"📚 Loaded metadata for {len(self.indexed_documents)} indexed documents")
            except Exception as e:
                logger.error(f"❌ Error loading index metadata: {e}")
                self.indexed_documents = {}
    
    def save_index_metadata(self):
        """Save metadata about indexed documents."""
        metadata_file = os.path.join(self.db_path, "index_metadata.json")
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.indexed_documents, f, indent=2, default=str)
            logger.info("💾 Saved index metadata")
        except Exception as e:
            logger.error(f"❌ Error saving index metadata: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for paragraph breaks
                    para_break = text.rfind('\n\n', start, end)
                    if para_break > start + chunk_size // 2:
                        end = para_break + 2
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_metadata_from_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": os.path.splitext(file_path)[1],
            "indexed_at": datetime.now().isoformat()
        }
        
        # Extract title from markdown headers
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Detect content type based on keywords
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["agent", "analyst", "specialist"]):
            metadata["content_type"] = "agent_documentation"
            
        if any(word in content_lower for word in ["tool", "function", "analysis", "calculation"]):
            metadata["content_type"] = "tool_documentation"
            
        if any(word in content_lower for word in ["workflow", "langgraph", "crewai", "flow"]):
            metadata["content_type"] = "workflow_documentation"
            
        if any(word in content_lower for word in ["implementation", "guide", "setup", "installation"]):
            metadata["content_type"] = "implementation_guide"
            
        if any(word in content_lower for word in ["optimization", "performance", "token", "cache"]):
            metadata["content_type"] = "optimization_guide"
        
        # Extract key features/capabilities
        features = []
        feature_patterns = [
            r'[✅✓]\s*(.+)',  # Checkmark lists
            r'[-*]\s*\*\*(.+?)\*\*',  # Bold items in lists
            r'##\s+(.+)',  # H2 headers
            r'###\s+(.+)',  # H3 headers
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            features.extend([match.strip() for match in matches])
        
        if features:
            metadata["features"] = features[:10]  # Limit to top 10
        
        return metadata
    
    def index_document(self, file_path: str, force_reindex: bool = False) -> bool:
        """
        Index a single document into the knowledge base.
        
        Args:
            file_path: Path to the document to index
            force_reindex: Force reindexing even if file hasn't changed
            
        Returns:
            bool: True if document was indexed successfully
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️  File not found: {file_path}")
                return False
            
            # Check if file has changed
            current_hash = self._get_file_hash(file_path)
            if not force_reindex and file_path in self.indexed_documents:
                if self.indexed_documents[file_path].get("hash") == current_hash:
                    logger.info(f"📄 File unchanged, skipping: {os.path.basename(file_path)}")
                    return True
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            if not content.strip():
                logger.warning(f"⚠️  Empty file: {file_path}")
                return False
            
            # Extract metadata
            metadata = self._extract_metadata_from_content(content, file_path)
            
            # Chunk the content
            chunks = self._chunk_text(content)
            
            # Index chunks
            indexed_chunks = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path}_{i}"
                
                # Create document chunk
                doc_chunk = DocumentChunk(
                    content=chunk,
                    source_file=file_path,
                    chunk_id=chunk_id,
                    metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)}
                )
                
                # Add to vector store if available
                if self.collection is not None:
                    try:
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[doc_chunk.metadata],
                            ids=[chunk_id]
                        )
                        indexed_chunks += 1
                    except Exception as e:
                        logger.error(f"❌ Error adding chunk to vector store: {e}")
                        continue
            
            # Update index metadata
            self.indexed_documents[file_path] = {
                "hash": current_hash,
                "indexed_at": datetime.now().isoformat(),
                "chunks": len(chunks),
                "indexed_chunks": indexed_chunks,
                "metadata": metadata
            }
            
            logger.info(f"✅ Indexed {os.path.basename(file_path)}: {indexed_chunks}/{len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error indexing document {file_path}: {e}")
            return False
    
    def index_directory(self, directory_path: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Index all files in a directory matching the given patterns.
        
        Args:
            directory_path: Path to directory to index
            file_patterns: List of file patterns to match (e.g., ['*.md', '*.py'])
            
        Returns:
            Dict with indexing results
        """
        if file_patterns is None:
            file_patterns = ['*.md', '*.py', '*.yaml', '*.yml', '*.txt']
        
        results = {
            "total_files": 0,
            "indexed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "files_processed": []
        }
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.warning(f"⚠️  Directory not found: {directory_path}")
                return results
            
            # Find all matching files
            all_files = []
            for pattern in file_patterns:
                all_files.extend(directory.rglob(pattern))
            
            results["total_files"] = len(all_files)
            
            for file_path in all_files:
                file_str = str(file_path)
                
                # Skip certain directories/files
                if any(skip in file_str for skip in ['.git', '__pycache__', '.pyc', 'node_modules']):
                    results["skipped_files"] += 1
                    continue
                
                success = self.index_document(file_str)
                
                if success:
                    results["indexed_files"] += 1
                    results["files_processed"].append({
                        "file": os.path.basename(file_str),
                        "status": "indexed"
                    })
                else:
                    results["failed_files"] += 1
                    results["files_processed"].append({
                        "file": os.path.basename(file_str),
                        "status": "failed"
                    })
            
            logger.info(f"📚 Directory indexing complete: {results['indexed_files']}/{results['total_files']} files indexed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error indexing directory {directory_path}: {e}")
            results["error"] = str(e)
            return results
    
    def search_knowledge(self, query: str, limit: int = 5, 
                        content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            content_type: Filter by content type (optional)
            
        Returns:
            List of relevant knowledge chunks
        """
        try:
            if self.collection is None:
                logger.warning("⚠️  Vector store not available, using fallback search")
                return self._fallback_search(query, limit, content_type)
            
            # Prepare search filters
            where_filter = {}
            if content_type:
                where_filter["content_type"] = content_type
            
            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter if where_filter else None
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "score": 1.0 - results['distances'][0][i] if results['distances'] else 0.8,
                        "id": results['ids'][0][i] if results['ids'] else f"result_{i}"
                    }
                    formatted_results.append(result)
            
            logger.info(f"🔍 Found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching knowledge base: {e}")
            return self._fallback_search(query, limit, content_type)
    
    def _fallback_search(self, query: str, limit: int, content_type: Optional[str]) -> List[Dict[str, Any]]:
        """Fallback search using simple text matching."""
        results = []
        query_lower = query.lower()
        
        for file_path, doc_info in self.indexed_documents.items():
            metadata = doc_info.get("metadata", {})
            
            # Filter by content type if specified
            if content_type and metadata.get("content_type") != content_type:
                continue
            
            # Simple text matching
            if any(term in file_path.lower() for term in query_lower.split()):
                results.append({
                    "content": f"Document: {os.path.basename(file_path)}",
                    "metadata": metadata,
                    "score": 0.7,
                    "id": file_path
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_agent_information(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available agents.
        
        Args:
            agent_name: Specific agent name to search for (optional)
            
        Returns:
            Dictionary with agent information
        """
        if agent_name:
            query = f"agent {agent_name} role capabilities tools"
        else:
            query = "agents roles capabilities marketing research analyst"
        
        results = self.search_knowledge(query, limit=10, content_type="agent_documentation")
        
        # Also search configuration files
        config_results = self.search_knowledge("agents.yaml configuration", limit=5)
        
        return {
            "agent_docs": results,
            "config_info": config_results,
            "total_results": len(results) + len(config_results)
        }
    
    def get_tool_information(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available tools.
        
        Args:
            tool_name: Specific tool name to search for (optional)
            
        Returns:
            Dictionary with tool information
        """
        if tool_name:
            query = f"tool {tool_name} function analysis parameters"
        else:
            query = "tools functions analysis profitability forecast market"
        
        results = self.search_knowledge(query, limit=10, content_type="tool_documentation")
        
        # Also search for advanced tools
        tool_results = self.search_knowledge("advanced_tools.py beverage_market_analysis", limit=5)
        
        return {
            "tool_docs": results,
            "implementation_info": tool_results,
            "total_results": len(results) + len(tool_results)
        }
    
    def get_workflow_information(self, workflow_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available workflows.
        
        Args:
            workflow_type: Specific workflow type (langgraph, crewai, etc.)
            
        Returns:
            Dictionary with workflow information
        """
        if workflow_type:
            query = f"workflow {workflow_type} implementation execution"
        else:
            query = "workflow langgraph crewai execution agents tasks"
        
        results = self.search_knowledge(query, limit=10, content_type="workflow_documentation")
        
        return {
            "workflow_docs": results,
            "total_results": len(results)
        }
    
    def get_feature_capabilities(self) -> Dict[str, Any]:
        """
        Get comprehensive information about system features and capabilities.
        
        Returns:
            Dictionary with feature information
        """
        # Search for different types of capabilities
        searches = {
            "optimization": self.search_knowledge("optimization performance token tracking", limit=5),
            "analysis": self.search_knowledge("analysis market research competitive", limit=5),
            "dashboard": self.search_knowledge("dashboard streamlit interface features", limit=5),
            "integration": self.search_knowledge("integration mem0 memory blackboard", limit=5),
            "tools": self.search_knowledge("tools advanced analysis functions", limit=5)
        }
        
        return {
            "capabilities": searches,
            "total_results": sum(len(results) for results in searches.values())
        }
    
    def build_knowledge_base(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build the complete knowledge base from specified files and directories.
        
        Args:
            force_rebuild: Force rebuilding even if files haven't changed
            
        Returns:
            Dictionary with build results
        """
        logger.info("🚀 Building Marketing Research Tool Knowledge Base...")
        
        # Define files and directories to index
        files_to_index = [
            "README_LANGGRAPH_DASHBOARD.md",
            "TOOL_RETRIEVAL_ANALYSIS_AND_RECOMMENDATIONS.md",
            "TOOL_USAGE_INSTRUCTIONS_FIX.md",
            "OPTIMIZATION_IMPLEMENTATION_COMPLETE.md",
            "LANGGRAPH_WORKFLOW_STATE_FIXES_COMPLETE.md",
            "LANGGRAPH_OPTIMIZATION_STRATEGIES_COMPLETE.md",
            "LANGGRAPH_IMPLEMENTATION_COMPLETE.md",
            "HYBRID_TOOL_SELECTION_IMPLEMENTATION_GUIDE.md",
            "ENHANCED_INTEGRATION_COMPLETE.md",
            "ENHANCED_AGENT_SYSTEM_IMPLEMENTATION_SUMMARY.md",
            "DEPENDENCY_MANAGEMENT_ENHANCEMENT_COMPLETE.md",
            "CONTEXTUAL_ENGINEERING_ALIGNMENT_ANALYSIS.md",
            "CONTEXT_ISOLATION_IMPLEMENTATION_COMPLETE.md",
            "AGENT_DATA_CACHING_AND_SHARING_SUMMARY.md",
            "ACCURATE_TOKEN_TRACKING_IMPLEMENTATION_COMPLETE.md",
            "langgraph_dashboard.py"
        ]
        
        directories_to_index = [
            "src/marketing_research_swarm/blackboard",
            "src/marketing_research_swarm/config",
            "src/marketing_research_swarm/context",
            "src/marketing_research_swarm/langgraph_workflow",
            "src/marketing_research_swarm/utils"
        ]
        
        specific_files = [
            "src/marketing_research_swarm/tools/advanced_tools_fixed.py"
        ]
        
        build_results = {
            "start_time": datetime.now().isoformat(),
            "files_indexed": 0,
            "directories_indexed": 0,
            "total_chunks": 0,
            "errors": [],
            "summary": {}
        }
        
        try:
            # Index individual documentation files
            for file_path in files_to_index:
                if os.path.exists(file_path):
                    success = self.index_document(file_path, force_reindex=force_rebuild)
                    if success:
                        build_results["files_indexed"] += 1
                        chunks = self.indexed_documents.get(file_path, {}).get("chunks", 0)
                        build_results["total_chunks"] += chunks
                    else:
                        build_results["errors"].append(f"Failed to index: {file_path}")
                else:
                    build_results["errors"].append(f"File not found: {file_path}")
            
            # Index specific files
            for file_path in specific_files:
                if os.path.exists(file_path):
                    success = self.index_document(file_path, force_reindex=force_rebuild)
                    if success:
                        build_results["files_indexed"] += 1
                        chunks = self.indexed_documents.get(file_path, {}).get("chunks", 0)
                        build_results["total_chunks"] += chunks
                    else:
                        build_results["errors"].append(f"Failed to index: {file_path}")
            
            # Index directories
            for dir_path in directories_to_index:
                if os.path.exists(dir_path):
                    dir_results = self.index_directory(dir_path)
                    build_results["directories_indexed"] += 1
                    build_results["files_indexed"] += dir_results["indexed_files"]
                    
                    # Estimate chunks from indexed files
                    for file_info in dir_results["files_processed"]:
                        if file_info["status"] == "indexed":
                            # Estimate chunks (actual count would require file reading)
                            build_results["total_chunks"] += 3  # Average estimate
                else:
                    build_results["errors"].append(f"Directory not found: {dir_path}")
            
            # Save metadata
            self.save_index_metadata()
            
            # Create summary
            build_results["summary"] = {
                "total_documents": len(self.indexed_documents),
                "documentation_files": len([f for f in files_to_index if os.path.exists(f)]),
                "source_directories": len([d for d in directories_to_index if os.path.exists(d)]),
                "knowledge_categories": len(self.knowledge_categories),
                "vector_store_available": self.collection is not None
            }
            
            build_results["end_time"] = datetime.now().isoformat()
            build_results["success"] = True
            
            logger.info(f"✅ Knowledge base built successfully!")
            logger.info(f"📊 Indexed {build_results['files_indexed']} files with {build_results['total_chunks']} chunks")
            
            return build_results
            
        except Exception as e:
            logger.error(f"❌ Error building knowledge base: {e}")
            build_results["error"] = str(e)
            build_results["success"] = False
            return build_results
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            stats = {
                "total_documents": len(self.indexed_documents),
                "vector_store_available": self.collection is not None,
                "db_path": self.db_path,
                "collection_name": self.collection_name,
                "last_updated": None,
                "content_types": {},
                "total_chunks": 0
            }
            
            # Analyze indexed documents
            for file_path, doc_info in self.indexed_documents.items():
                metadata = doc_info.get("metadata", {})
                content_type = metadata.get("content_type", "unknown")
                
                stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
                stats["total_chunks"] += doc_info.get("chunks", 0)
                
                # Track latest update
                indexed_at = doc_info.get("indexed_at")
                if indexed_at and (not stats["last_updated"] or indexed_at > stats["last_updated"]):
                    stats["last_updated"] = indexed_at
            
            # Get vector store stats if available
            if self.collection is not None:
                try:
                    collection_count = self.collection.count()
                    stats["vector_store_documents"] = collection_count
                except Exception as e:
                    stats["vector_store_error"] = str(e)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting knowledge stats: {e}")
            return {"error": str(e)}

# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> MarketingResearchKnowledgeBase:
    """Get the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = MarketingResearchKnowledgeBase()
    return _knowledge_base

def initialize_knowledge_base(force_rebuild: bool = False) -> Dict[str, Any]:
    """Initialize and build the knowledge base."""
    kb = get_knowledge_base()
    return kb.build_knowledge_base(force_rebuild=force_rebuild)