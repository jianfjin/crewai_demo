"""
Enhanced LangSmith Monitor for LangGraph Workflows
"""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client as LangSmithClient
    langsmith_client = LangSmithClient()
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    langsmith_client = None


class EnhancedLangSmithMonitor:
    """Enhanced LangSmith monitoring with proper UUID handling."""
    
    def __init__(self):
        self.client = None
        self.project_name = "marketing-research-swarm"
        self.available = False
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize LangSmith client with proper error handling."""
        try:
            if not LANGSMITH_AVAILABLE:
                return
                
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if not api_key:
                logger.warning("LANGCHAIN_API_KEY not found")
                return
                
            self.client = langsmith_client
            self.available = True
            logger.info(f"✅ Enhanced LangSmith monitoring enabled for project: {self.project_name}")
                
        except Exception as e:
            logger.warning(f"Enhanced LangSmith initialization failed: {e}")
            self.available = False
            
    def create_run_tracer(self, workflow_id: str):
        """Create a tracer for a specific workflow run."""
        if not self.available:
            return None
            
        try:
            # Set environment variables for this run
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            logger.info(f"🔍 Created LangSmith tracer for workflow: {workflow_id}")
            # Note: This would need to import create_langsmith_tracer from utils
            return None  # Placeholder for now
            
        except Exception as e:
            logger.error(f"Failed to create LangSmith tracer: {e}")
            return None
            
    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent runs from LangSmith."""
        if not self.available:
            return []
            
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit
            ))
            
            formatted_runs = []
            for run in runs:
                formatted_runs.append({
                    'id': str(run.id),
                    'name': run.name or 'Unknown',
                    'status': run.status or 'unknown',
                    'start_time': run.start_time,
                    'end_time': run.end_time,
                    'total_tokens': getattr(run, 'total_tokens', 0),
                    'tags': getattr(run, 'tags', []),
                    'url': f"https://smith.langchain.com/public/{run.id}/r"
                })
                
            return formatted_runs
            
        except Exception as e:
            logger.error(f"Failed to get LangSmith runs: {e}")
            return []