"""
RAG (Retrieval-Augmented Generation) Module

This module provides intelligent knowledge retrieval and chat capabilities
for the Marketing Research Tool.

Components:
- knowledge_base.py: Core RAG implementation with vector storage
- chat_integration.py: Integration with chat agents
- query_processor.py: Query understanding and routing
"""

from .knowledge_base import (
    MarketingResearchKnowledgeBase,
    get_knowledge_base,
    initialize_knowledge_base
)

__all__ = [
    'MarketingResearchKnowledgeBase',
    'get_knowledge_base', 
    'initialize_knowledge_base'
]