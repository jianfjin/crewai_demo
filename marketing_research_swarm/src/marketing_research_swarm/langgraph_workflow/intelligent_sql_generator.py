"""
Intelligent SQL Generator using LLM for Text-to-SQL

This module converts natural language queries to SQL using LLM with data context awareness.
"""

import logging
import duckdb
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from langchain.schema import SystemMessage, HumanMessage
from .data_context_manager import get_data_context_manager

logger = logging.getLogger(__name__)

class IntelligentSQLGenerator:
    """Generates SQL queries from natural language using LLM with data context."""
    
    def __init__(self, llm):
        self.llm = llm
        self.data_context_manager = get_data_context_manager()
        
    def generate_sql_from_query(self, user_query: str, data_source: str = "beverage_sales") -> Dict[str, Any]:
        """Generate SQL query from natural language with data context awareness."""
        
        try:
            # Get data context
            data_context = self.data_context_manager.get_data_context(data_source)
            
            if "error" in data_context:
                return {"error": f"Data context error: {data_context['error']}"}
            
            # Generate SQL using LLM
            sql_query = self._generate_sql_with_llm(user_query, data_context)
            
            if not sql_query or "error" in sql_query:
                return {"error": "Failed to generate SQL query"}
            
            # Validate and execute SQL
            result = self._execute_and_validate_sql(sql_query["query"], data_source)
            
            return {
                "query": sql_query["query"],
                "explanation": sql_query.get("explanation", ""),
                "result": result,
                "data_context_used": True
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent SQL generation: {e}")
            return {"error": str(e)}
    
    def _generate_sql_with_llm(self, user_query: str, data_context: Dict[str, Any]) -> Dict[str, str]:
        """Use LLM to generate SQL query with full data context."""
        
        schema = data_context["schema"]
        preview = data_context["preview"]
        distinct_values = data_context["distinct_values"]
        
        # Create comprehensive prompt with data context
        system_prompt = """You are an expert SQL analyst with deep knowledge of data analysis and business intelligence. 

Your task is to convert natural language questions into precise SQL queries using DuckDB syntax.

CRITICAL REQUIREMENTS:
1. Use ONLY the columns and table structure provided in the data context
2. Generate syntactically correct DuckDB SQL
3. Focus on answering the specific question asked
4. Use appropriate aggregations, filters, and sorting
5. Include meaningful column aliases
6. Handle edge cases (nulls, empty results, etc.)
7. Return results in a logical order

RESPONSE FORMAT:
Return a JSON object with:
- "query": The SQL query string
- "explanation": Brief explanation of what the query does
"""

        # Build detailed data context for the prompt
        context_prompt = f"""
## DATA CONTEXT

**Table Name:** beverage_sales

**Schema ({schema['shape']['rows']:,} rows, {schema['shape']['columns']} columns):**
"""
        
        for col_info in schema["columns"]:
            context_prompt += f"""
- **{col_info['name']}** ({col_info['dtype']})
  - Unique values: {col_info['unique_count']:,}
  - Null values: {col_info['null_count']:,} ({col_info['null_percentage']:.1f}%)"""
            
            if col_info.get('min') is not None:
                context_prompt += f"\n  - Range: {col_info['min']} to {col_info['max']} (avg: {col_info['mean']:.2f})"
            
            if col_info['sample_values']:
                context_prompt += f"\n  - Examples: {', '.join(map(str, col_info['sample_values']))}"
            
            context_prompt += "\n"
        
        # Add distinct values for key categorical columns
        context_prompt += "\n**Key Categorical Values:**\n"
        for col, values in distinct_values.items():
            if len(values) <= 20:  # Only show columns with reasonable number of values
                context_prompt += f"- **{col}:** {', '.join(map(str, values))}\n"
        
        # Add sample data
        context_prompt += f"\n**Sample Data (first 3 rows):**\n"
        for i, row in enumerate(preview["head"][:3], 1):
            context_prompt += f"Row {i}: {row}\n"
        
        # Add the user query
        query_prompt = f"""
## USER QUESTION
"{user_query}"

## TASK
Generate a SQL query that answers this question using the beverage_sales table.

IMPORTANT GUIDELINES:
- Use exact column names from the schema above
- Consider the data types and value ranges
- Use appropriate WHERE clauses based on available categorical values
- Include proper aggregations if the question asks for summaries
- Sort results in a meaningful way
- Handle potential null values appropriately

Generate the SQL query as a JSON response with "query" and "explanation" fields.
"""
        
        try:
            system_message = SystemMessage(content=system_prompt)
            human_message = HumanMessage(content=context_prompt + query_prompt)
            
            response = self.llm.invoke([system_message, human_message])
            
            # Parse JSON response
            import json
            try:
                # Try to extract JSON from response
                response_text = response.content.strip()
                
                # Handle cases where response might have markdown formatting
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                
                # Parse JSON
                result = json.loads(response_text)
                
                if "query" in result:
                    return result
                else:
                    # Fallback: treat entire response as query
                    return {
                        "query": response_text,
                        "explanation": "Generated SQL query"
                    }
                    
            except json.JSONDecodeError:
                # Fallback: treat response as raw SQL
                return {
                    "query": response.content.strip(),
                    "explanation": "Generated SQL query (raw response)"
                }
                
        except Exception as e:
            logger.error(f"Error generating SQL with LLM: {e}")
            return {"error": str(e)}
    
    def _execute_and_validate_sql(self, sql_query: str, data_source: str) -> Dict[str, Any]:
        """Execute and validate the generated SQL query."""
        
        try:
            # Get data
            df = self.data_context_manager.data_cache.get(data_source)
            if df is None or df.empty:
                return {"error": "No data available for query execution"}
            
            # Create DuckDB connection
            conn = duckdb.connect(':memory:')
            
            # Register the dataframe
            conn.register('beverage_sales', df)
            
            logger.info(f"🗃️ Executing intelligent SQL: {sql_query}")
            
            # Execute the query
            result = conn.execute(sql_query).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            # Format results
            if not result:
                return {
                    "success": True,
                    "rows": 0,
                    "message": "Query executed successfully but returned no results",
                    "data": []
                }
            
            # Convert to structured format with proper serialization
            formatted_data = []
            for row in result[:50]:  # Limit to 50 rows for performance
                row_dict = {}
                for col, value in zip(columns, row):
                    # Convert numpy types to native Python types for serialization
                    if hasattr(value, 'item'):  # numpy scalar
                        row_dict[col] = value.item()
                    elif hasattr(value, 'tolist'):  # numpy array
                        row_dict[col] = value.tolist()
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        row_dict[col] = value
                    else:
                        # Convert other types to string as fallback
                        row_dict[col] = str(value)
                formatted_data.append(row_dict)
            
            conn.close()
            
            return {
                "success": True,
                "rows": len(result),
                "columns": columns,
                "data": formatted_data,
                "limited": len(result) > 50,
                "total_rows": len(result)
            }
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return {"error": f"SQL execution error: {str(e)}"}
    
    def format_sql_results_for_analysis(self, sql_result: Dict[str, Any]) -> str:
        """Format SQL results for analysis by other agents."""
        
        if "error" in sql_result:
            return f"SQL Query Error: {sql_result['error']}"
        
        if not sql_result.get("success"):
            return "SQL query failed to execute"
        
        if sql_result["rows"] == 0:
            return "SQL query executed successfully but returned no results"
        
        # Format results for readability
        formatted_result = f"""
## 🗃️ SQL Query Results

**Query executed successfully**
- **Rows returned:** {sql_result['rows']:,}
- **Columns:** {', '.join(sql_result['columns'])}

### 📊 Data Results:
"""
        
        for i, row in enumerate(sql_result["data"][:20], 1):  # Show first 20 rows
            formatted_result += f"\n**Row {i}:**\n"
            for col, value in row.items():
                formatted_result += f"  - {col}: {value}\n"
        
        if sql_result.get("limited") and sql_result["total_rows"] > 20:
            formatted_result += f"\n... and {sql_result['total_rows'] - 20} more rows\n"
        
        # Add summary insights
        if sql_result["data"]:
            formatted_result += "\n### 💡 Key Insights:\n"
            
            # Identify key metrics
            first_row = sql_result["data"][0]
            numeric_cols = [col for col, val in first_row.items() if isinstance(val, (int, float))]
            
            if numeric_cols:
                formatted_result += f"- Numeric metrics available: {', '.join(numeric_cols)}\n"
            
            # Identify categorical groupings
            categorical_cols = [col for col, val in first_row.items() if isinstance(val, str)]
            if categorical_cols:
                formatted_result += f"- Categorical dimensions: {', '.join(categorical_cols)}\n"
            
            # Show top performers if applicable
            if len(sql_result["data"]) > 1 and numeric_cols:
                top_metric = numeric_cols[0]
                top_row = sql_result["data"][0]
                formatted_result += f"- Top performer by {top_metric}: {top_row.get(top_metric)}\n"
        
        return formatted_result

# Global instance
_intelligent_sql_generator = None

def get_intelligent_sql_generator(llm) -> IntelligentSQLGenerator:
    """Get intelligent SQL generator instance."""
    global _intelligent_sql_generator
    if _intelligent_sql_generator is None:
        _intelligent_sql_generator = IntelligentSQLGenerator(llm)
    return _intelligent_sql_generator