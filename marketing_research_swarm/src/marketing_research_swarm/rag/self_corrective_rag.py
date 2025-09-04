"""
Self-Corrective RAG Implementation with Hallucination Detection and Web Search Fallback

This module implements a self-corrective RAG system that:
1. Retrieves relevant information from the knowledge base
2. Checks for hallucinations in generated answers
3. Grades answer quality and helpfulness
4. Falls back to web search when knowledge base is insufficient
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import re
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SelfCorrectiveRAG:
    """
    Self-corrective RAG system with hallucination detection and web search fallback.
    """
    
    def __init__(self, knowledge_base, llm_model="gpt-4o-mini"):
        """Initialize the self-corrective RAG system."""
        self.knowledge_base = knowledge_base
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        
        # Initialize web search tool with error handling
        try:
            self.web_search_tool = TavilySearchResults(max_results=3)
        except Exception as e:
            logger.warning(f"Web search tool not available: {e}")
            self.web_search_tool = None
        
        # Retry limits
        self.max_retrieval_retries = 2
        self.max_generation_retries = 2
        
    def process_query_with_correction(self, query: str) -> Dict[str, Any]:
        """
        Process a query with self-correction mechanisms.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary with corrected response information
        """
        try:
            # Step 1: Initial retrieval from knowledge base
            retrieval_result = self._retrieve_with_retry(query)
            
            if retrieval_result["has_relevant_info"]:
                # Step 2: Generate answer using RAG
                generation_result = self._generate_answer_with_correction(
                    query, retrieval_result["documents"]
                )
                
                if generation_result["is_valid"]:
                    return {
                        "query": query,
                        "answer": generation_result["answer"],
                        "source": "knowledge_base",
                        "documents": retrieval_result["documents"],
                        "confidence": generation_result["confidence"],
                        "corrections_made": generation_result["corrections_made"],
                        "retrieval_attempts": retrieval_result["attempts"],
                        "generation_attempts": generation_result["attempts"]
                    }
            
            # Step 3: Fallback to web search
            logger.info(f"Knowledge base insufficient for query: {query}")
            web_result = self._web_search_fallback(query)
            
            return {
                "query": query,
                "answer": web_result["answer"],
                "source": "web_search",
                "web_results": web_result["results"],
                "confidence": web_result["confidence"],
                "corrections_made": 0,
                "retrieval_attempts": retrieval_result["attempts"],
                "generation_attempts": 0
            }
            
        except Exception as e:
            logger.error(f"Error in self-corrective RAG: {e}")
            return {
                "query": query,
                "answer": "I encountered an error processing your request. Please try rephrasing your question.",
                "source": "error",
                "confidence": 0.1,
                "error": str(e)
            }
    
    def _retrieve_with_retry(self, query: str, attempt: int = 1) -> Dict[str, Any]:
        """
        Retrieve documents with retry mechanism for query modification.
        """
        try:
            # Initial search
            documents = self.knowledge_base.search_knowledge(query, limit=5)
            
            # Check if we have relevant documents
            if documents and len(documents) > 0:
                # Check relevance scores
                avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents)
                if avg_score > 0.3:  # Threshold for relevance
                    return {
                        "has_relevant_info": True,
                        "documents": documents,
                        "attempts": attempt
                    }
            
            # If no relevant documents and we can retry
            if attempt < self.max_retrieval_retries:
                # Modify query for better retrieval
                modified_query = self._modify_query_for_retrieval(query, attempt)
                logger.info(f"Retrying retrieval with modified query: {modified_query}")
                return self._retrieve_with_retry(modified_query, attempt + 1)
            
            return {
                "has_relevant_info": False,
                "documents": documents or [],
                "attempts": attempt
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval attempt {attempt}: {e}")
            return {
                "has_relevant_info": False,
                "documents": [],
                "attempts": attempt
            }
    
    def _modify_query_for_retrieval(self, query: str, attempt: int) -> str:
        """
        Modify query to improve retrieval on retry.
        """
        if attempt == 1:
            # First retry: Add marketing research context
            return f"marketing research {query} analysis tools agents"
        elif attempt == 2:
            # Second retry: Extract key terms
            key_terms = self._extract_key_terms(query)
            return " ".join(key_terms)
        
        return query
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better retrieval."""
        # Remove common words and extract meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "which", "when", "where", "why", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "about"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _generate_answer_with_correction(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """
        Generate answer with hallucination detection and correction.
        """
        attempt = 1
        
        while attempt <= self.max_generation_retries:
            try:
                # Generate initial answer
                answer = self._generate_answer(query, documents)
                
                # Step 1: Hallucination grader
                hallucination_result = self._grade_hallucination(answer, documents)
                
                if not hallucination_result["is_grounded"]:
                    logger.warning(f"Hallucination detected in attempt {attempt}")
                    if attempt < self.max_generation_retries:
                        # Retry with more explicit grounding instructions
                        attempt += 1
                        continue
                    else:
                        # Use fallback answer
                        answer = self._generate_grounded_fallback(query, documents)
                
                # Step 2: Answer grader
                answer_quality = self._grade_answer_quality(query, answer)
                
                if not answer_quality["is_helpful"]:
                    logger.warning(f"Answer quality insufficient in attempt {attempt}")
                    if attempt < self.max_generation_retries:
                        # Rewrite query and retry
                        modified_query = self._rewrite_query_for_generation(query, attempt)
                        answer = self._generate_answer(modified_query, documents)
                        attempt += 1
                        continue
                
                return {
                    "is_valid": True,
                    "answer": answer,
                    "confidence": min(hallucination_result["confidence"], answer_quality["confidence"]),
                    "corrections_made": attempt - 1,
                    "attempts": attempt,
                    "hallucination_check": hallucination_result,
                    "quality_check": answer_quality
                }
                
            except Exception as e:
                logger.error(f"Error in generation attempt {attempt}: {e}")
                attempt += 1
        
        # If all attempts failed, return fallback
        return {
            "is_valid": False,
            "answer": "I couldn't generate a reliable answer from the available information.",
            "confidence": 0.2,
            "corrections_made": self.max_generation_retries,
            "attempts": self.max_generation_retries
        }
    
    def _generate_answer(self, query: str, documents: List[Dict]) -> str:
        """Generate answer based on retrieved documents."""
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("file_name", f"Document {i}")
            context_parts.append(f"Source {i} ({source}): {content}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a helpful marketing research assistant. Answer the user's question based ONLY on the provided context documents. 

IMPORTANT RULES:
1. Only use information that is explicitly stated in the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Do not add information from your general knowledge
4. Cite which source(s) you're using for each piece of information
5. Be specific and helpful while staying grounded in the provided context

Context Documents:
{context}

Answer the question based only on the above context."""

        user_prompt = f"Question: {query}\n\nPlease provide a helpful answer based only on the provided context documents."
        
        try:
            messages = [
                SystemMessage(content=system_prompt.format(context=context)),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error generating an answer from the available information."
    
    def _grade_hallucination(self, answer: str, documents: List[Dict]) -> Dict[str, Any]:
        """
        Grade whether the answer contains hallucinations (information not in documents).
        """
        # Prepare document content for checking
        doc_content = "\n".join([doc.get("content", "") for doc in documents])
        
        system_prompt = """You are a grader assessing whether an answer is grounded in the provided documents.

Your task is to determine if the answer contains any information that is NOT explicitly stated or reasonably inferred from the provided documents.

Respond with a JSON object containing:
- "is_grounded": boolean (true if answer is fully grounded in documents)
- "confidence": float (0.0 to 1.0)
- "issues": list of specific hallucination issues found (empty if none)
- "explanation": brief explanation of your assessment

Documents:
{documents}

Answer to evaluate:
{answer}"""

        try:
            messages = [
                SystemMessage(content=system_prompt.format(documents=doc_content, answer=answer)),
                HumanMessage(content="Please evaluate if this answer is grounded in the provided documents.")
            ]
            
            response = self.llm.invoke(messages)
            result = json.loads(response.content.strip())
            
            return {
                "is_grounded": result.get("is_grounded", False),
                "confidence": result.get("confidence", 0.5),
                "issues": result.get("issues", []),
                "explanation": result.get("explanation", "")
            }
            
        except Exception as e:
            logger.error(f"Error in hallucination grading: {e}")
            return {
                "is_grounded": False,
                "confidence": 0.3,
                "issues": ["Error in hallucination detection"],
                "explanation": f"Error occurred: {str(e)}"
            }
    
    def _grade_answer_quality(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Grade whether the answer is helpful and addresses the user's question.
        """
        system_prompt = """You are a grader assessing whether an answer is helpful and addresses the user's question.

Evaluate the answer based on:
1. Does it directly address the question asked?
2. Is it specific and actionable?
3. Is it clear and well-structured?
4. Does it provide useful information for the user's needs?

Respond with a JSON object containing:
- "is_helpful": boolean (true if answer is helpful and addresses the question)
- "confidence": float (0.0 to 1.0)
- "strengths": list of positive aspects
- "weaknesses": list of areas for improvement
- "explanation": brief explanation of your assessment

Question: {query}
Answer: {answer}"""

        try:
            messages = [
                SystemMessage(content=system_prompt.format(query=query, answer=answer)),
                HumanMessage(content="Please evaluate if this answer is helpful and addresses the question.")
            ]
            
            response = self.llm.invoke(messages)
            result = json.loads(response.content.strip())
            
            return {
                "is_helpful": result.get("is_helpful", False),
                "confidence": result.get("confidence", 0.5),
                "strengths": result.get("strengths", []),
                "weaknesses": result.get("weaknesses", []),
                "explanation": result.get("explanation", "")
            }
            
        except Exception as e:
            logger.error(f"Error in answer quality grading: {e}")
            return {
                "is_helpful": False,
                "confidence": 0.3,
                "strengths": [],
                "weaknesses": ["Error in quality assessment"],
                "explanation": f"Error occurred: {str(e)}"
            }
    
    def _rewrite_query_for_generation(self, query: str, attempt: int) -> str:
        """
        Rewrite query to improve answer generation.
        """
        if attempt == 1:
            return f"Please explain {query} in detail with specific examples"
        elif attempt == 2:
            return f"What are the key aspects of {query} that I should know about?"
        
        return query
    
    def _generate_grounded_fallback(self, query: str, documents: List[Dict]) -> str:
        """
        Generate a conservative, grounded fallback answer.
        """
        if not documents:
            return "I don't have enough information in my knowledge base to answer this question accurately."
        
        # Extract key information from documents
        doc_summaries = []
        for doc in documents[:2]:
            content = doc.get("content", "")[:200]  # First 200 chars
            metadata = doc.get("metadata", {})
            source = metadata.get("file_name", "document")
            doc_summaries.append(f"From {source}: {content}...")
        
        return f"Based on the available information:\n\n" + "\n\n".join(doc_summaries) + f"\n\nFor more specific information about '{query}', you may need to consult additional resources."
    
    def _web_search_fallback(self, query: str) -> Dict[str, Any]:
        """
        Perform web search when knowledge base is insufficient.
        """
        try:
            # Check if web search tool is available
            if not self.web_search_tool:
                return {
                    "answer": f"I couldn't find relevant information about '{query}' in my knowledge base, and web search is not available. Please try asking about topics related to marketing research tools, agents, workflows, or system features that are covered in my knowledge base.",
                    "results": [],
                    "confidence": 0.2
                }
            
            # Enhance query for marketing research context
            search_query = f"marketing research {query} analysis tools"
            
            # Perform web search
            search_results = self.web_search_tool.run(search_query)
            
            if not search_results:
                return {
                    "answer": f"I couldn't find relevant information about '{query}' in my knowledge base or through web search. Could you please provide more specific details or rephrase your question?",
                    "results": [],
                    "confidence": 0.2
                }
            
            # Generate answer from web search results
            answer = self._generate_answer_from_web_results(query, search_results)
            
            return {
                "answer": answer,
                "results": search_results,
                "confidence": 0.7  # Lower confidence for web results
            }
            
        except Exception as e:
            logger.error(f"Error in web search fallback: {e}")
            return {
                "answer": f"I couldn't find information about '{query}' in my knowledge base, and web search is currently unavailable. Please try rephrasing your question or ask about topics related to marketing research tools and agents.",
                "results": [],
                "confidence": 0.1
            }
    
    def _generate_answer_from_web_results(self, query: str, search_results: List[Dict]) -> str:
        """
        Generate answer from web search results.
        """
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            if isinstance(result, dict):
                title = result.get("title", f"Result {i}")
                content = result.get("content", result.get("snippet", ""))
                url = result.get("url", "")
                context_parts.append(f"Source {i} - {title}: {content}")
            else:
                context_parts.append(f"Source {i}: {str(result)}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a marketing research assistant. Answer the user's question based on the provided web search results.

IMPORTANT:
1. Focus on marketing research, analysis tools, and business intelligence aspects
2. Provide practical, actionable information
3. Mention that this information comes from web search
4. Be helpful while acknowledging the external source

Web Search Results:
{context}

Provide a helpful answer based on these search results."""

        try:
            messages = [
                SystemMessage(content=system_prompt.format(context=context)),
                HumanMessage(content=f"Question: {query}")
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            # Add web search disclaimer
            answer += "\n\n*Note: This information was found through web search as it wasn't available in my knowledge base.*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer from web results: {e}")
            return f"I found some web search results for '{query}', but encountered an error processing them. Please try rephrasing your question."