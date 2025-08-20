"""
StateGraph visualization for the dashboard.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None


class StateGraphVisualizer:
    """StateGraph visualization for the dashboard."""
    
    def __init__(self):
        self.available = PLOTLY_AVAILABLE
        
        # Define actual agent dependencies from the LangGraph workflow
        self.agent_dependencies = {
            "market_research_analyst": [],  # Can run first
            "competitive_analyst": [],  # Can run first
            "data_analyst": [],  # Can run first
            "content_strategist": ["market_research_analyst"],  # Needs market research
            "creative_copywriter": ["content_strategist"],  # Needs content strategy
            "brand_performance_specialist": ["competitive_analyst", "data_analyst"],  # Needs competitive and data analysis
            "forecasting_specialist": ["market_research_analyst", "data_analyst"],  # Needs market research and data analysis
            "campaign_optimizer": ["data_analyst", "content_strategist"],  # Needs data and content strategy
        }
        
    def get_execution_order(self, selected_agents: List[str]) -> List[List[str]]:
        """Calculate the actual execution order based on dependencies."""
        if not selected_agents:
            return []
        
        # Create execution layers
        execution_layers = []
        remaining_agents = set(selected_agents)
        completed_agents = set()
        
        while remaining_agents:
            current_layer = []
            
            # Find agents whose dependencies are satisfied
            for agent in list(remaining_agents):
                dependencies = self.agent_dependencies.get(agent, [])
                
                # Check if all dependencies are completed or not in selected agents
                dependencies_met = all(
                    dep in completed_agents or dep not in selected_agents
                    for dep in dependencies
                )
                
                if dependencies_met:
                    current_layer.append(agent)
            
            if not current_layer:
                # Circular dependency or error - add remaining agents
                current_layer = list(remaining_agents)
            
            execution_layers.append(current_layer)
            
            # Update completed and remaining
            for agent in current_layer:
                completed_agents.add(agent)
                remaining_agents.discard(agent)
        
        return execution_layers
    
    def draw_ascii_graph(self, selected_agents: List[str]) -> str:
        """Create ASCII representation of the workflow graph."""
        if not selected_agents:
            return "No agents selected"
        
        execution_order = self.get_execution_order(selected_agents)
        
        ascii_graph = []
        ascii_graph.append("LangGraph Workflow Execution Order:")
        ascii_graph.append("=" * 50)
        ascii_graph.append("")
        ascii_graph.append("┌─────────────┐")
        ascii_graph.append("│    START    │")
        ascii_graph.append("└─────────────┘")
        ascii_graph.append("       │")
        ascii_graph.append("       ▼")
        ascii_graph.append("┌─────────────┐")
        ascii_graph.append("│   CONTEXT   │")
        ascii_graph.append("│ OPTIMIZATION│")
        ascii_graph.append("└─────────────┘")
        ascii_graph.append("       │")
        ascii_graph.append("       ▼")
        ascii_graph.append("┌─────────────┐")
        ascii_graph.append("│    AGENT    │")
        ascii_graph.append("│   ROUTER    │")
        ascii_graph.append("└─────────────┘")
        
        # Add execution layers
        for layer_idx, layer in enumerate(execution_order):
            ascii_graph.append("       │")
            ascii_graph.append("       ▼")
            ascii_graph.append(f"  Layer {layer_idx + 1}:")
            
            if len(layer) == 1:
                agent = layer[0]
                agent_display = agent.replace('_', ' ').title()
                # Use wider box for full names
                box_width = max(len(agent_display) + 4, 20)
                ascii_graph.append("┌" + "─" * box_width + "┐")
                ascii_graph.append(f"│{agent_display.center(box_width)}│")
                ascii_graph.append("└" + "─" * box_width + "┘")
            else:
                # Multiple agents in parallel
                ascii_graph.append("       │")
                if len(layer) == 2:
                    ascii_graph.append("   ┌───┴───┐")
                    ascii_graph.append("   ▼       ▼")
                    agent1_display = layer[0].replace('_', ' ').title()
                    agent2_display = layer[1].replace('_', ' ').title()
                    box_width = max(len(agent1_display) + 4, len(agent2_display) + 4, 20)
                    ascii_graph.append("┌" + "─" * box_width + "┐  ┌" + "─" * box_width + "┐")
                    ascii_graph.append(f"│{agent1_display.center(box_width)}│  │{agent2_display.center(box_width)}│")
                    ascii_graph.append("└" + "─" * box_width + "┘  └" + "─" * box_width + "┘")
                else:
                    # More than 2 agents - show them vertically
                    for agent in layer:
                        agent_display = agent.replace('_', ' ').title()
                        box_width = max(len(agent_display) + 4, 20)
                        ascii_graph.append("┌" + "─" * box_width + "┐")
                        ascii_graph.append(f"│{agent_display.center(box_width)}│")
                        ascii_graph.append("└" + "─" * box_width + "┘")
        
        ascii_graph.append("       │")
        ascii_graph.append("       ▼")
        ascii_graph.append("┌─────────────┐")
        ascii_graph.append("│   RESULT    │")
        ascii_graph.append("│ COMPRESSION │")
        ascii_graph.append("└─────────────┘")
        ascii_graph.append("       │")
        ascii_graph.append("       ▼")
        ascii_graph.append("┌─────────────┐")
        ascii_graph.append("│  FINALIZE   │")
        ascii_graph.append("└─────────────┘")
        ascii_graph.append("       │")
        ascii_graph.append("       ▼")
        ascii_graph.append("┌─────────────┐")
        ascii_graph.append("│     END     │")
        ascii_graph.append("└─────────────┘")
        
        return "\n".join(ascii_graph)
    
    def create_mermaid_graph(self, selected_agents: List[str]) -> str:
        """Create Mermaid diagram representation of the workflow."""
        if not selected_agents:
            return "graph TD\n    A[No agents selected]"
        
        execution_order = self.get_execution_order(selected_agents)
        
        mermaid_lines = ["graph TD"]
        mermaid_lines.append("    START([START]) --> CONTEXT[Context Optimization]")
        mermaid_lines.append("    CONTEXT --> ROUTER[Agent Router]")
        
        # Add agent nodes and dependencies
        for layer_idx, layer in enumerate(execution_order):
            for agent in layer:
                agent_id = agent.upper().replace("_", "")
                agent_name = agent.replace("_", " ").title()
                mermaid_lines.append(f"    {agent_id}[{agent_name}]")
                
                # Connect from router or dependencies
                dependencies = self.agent_dependencies.get(agent, [])
                connected_deps = [dep for dep in dependencies if dep in selected_agents]
                
                if not connected_deps:
                    mermaid_lines.append(f"    ROUTER --> {agent_id}")
                else:
                    for dep in connected_deps:
                        dep_id = dep.upper().replace("_", "")
                        mermaid_lines.append(f"    {dep_id} --> {agent_id}")
                
                # Connect to compression
                mermaid_lines.append(f"    {agent_id} --> COMPRESSION[Result Compression]")
        
        mermaid_lines.append("    COMPRESSION --> FINALIZE[Finalize]")
        mermaid_lines.append("    FINALIZE --> END([END])")
        
        # Add styling
        mermaid_lines.append("    classDef startEnd fill:#e1f5fe")
        mermaid_lines.append("    classDef process fill:#f3e5f5")
        mermaid_lines.append("    classDef agent fill:#e8f5e8")
        mermaid_lines.append("    class START,END startEnd")
        mermaid_lines.append("    class CONTEXT,ROUTER,COMPRESSION,FINALIZE process")
        
        for agent in selected_agents:
            agent_id = agent.upper().replace("_", "")
            mermaid_lines.append(f"    class {agent_id} agent")
        
        return "\n".join(mermaid_lines)
        
    def create_workflow_graph(self, selected_agents: List[str], analysis_type: str = "comprehensive"):
        """Create a visual representation of the workflow StateGraph."""
        if not self.available:
            return None
            
        try:
            # Create a simple workflow visualization using Plotly
            import plotly.graph_objects as go
            
            if not selected_agents:
                return None
            
            # Define node positions
            nodes = ["start", "context_optimization", "agent_router"] + selected_agents + ["result_compression", "finalize", "end"]
            
            # Create layout
            y_positions = {}
            y_positions["start"] = 4
            y_positions["context_optimization"] = 3
            y_positions["agent_router"] = 2
            
            # Position agents horizontally
            agent_count = len(selected_agents)
            if agent_count > 0:
                agent_spacing = 2.0 / max(agent_count - 1, 1) if agent_count > 1 else 0
                start_x = -1.0 if agent_count > 1 else 0
                
                for i, agent in enumerate(selected_agents):
                    y_positions[agent] = 1
            
            y_positions["result_compression"] = 0
            y_positions["finalize"] = -1
            y_positions["end"] = -2
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            
            for i, node in enumerate(nodes):
                if node in selected_agents:
                    x_pos = start_x + selected_agents.index(node) * agent_spacing if agent_count > 1 else 0
                else:
                    x_pos = 0
                    
                node_x.append(x_pos)
                node_y.append(y_positions.get(node, 0))
                node_text.append(node.replace('_', ' ').title())
                
                # Color coding and sizing
                if node == "start":
                    node_colors.append('#00ff00')  # Green
                    node_sizes.append(40)
                elif node == "end":
                    node_colors.append('#ff0000')  # Red
                    node_sizes.append(40)
                elif node in selected_agents:
                    node_colors.append('#1f77b4')  # Blue
                    node_sizes.append(35)
                else:
                    node_colors.append('#ff7f0e')  # Orange
                    node_sizes.append(30)
            
            # Create the figure
            fig = go.Figure()
            
            # Add edges first (so they appear behind nodes)
            edge_x = []
            edge_y = []
            
            # Create proper workflow connections with dependencies
            connections = [
                ("start", "context_optimization"),
                ("context_optimization", "agent_router")
            ]
            
            # Add connections based on actual dependencies
            for agent in selected_agents:
                dependencies = self.agent_dependencies.get(agent, [])
                connected_deps = [dep for dep in dependencies if dep in selected_agents]
                
                if not connected_deps:
                    # No dependencies - connect from router
                    connections.append(("agent_router", agent))
                else:
                    # Has dependencies - connect from dependency agents
                    for dep in connected_deps:
                        connections.append((dep, agent))
                
                # All agents connect to compression
                connections.append((agent, "result_compression"))
            
            # Add final connections
            connections.extend([
                ("result_compression", "finalize"),
                ("finalize", "end")
            ])
            
            # Create edge coordinates
            for start_node, end_node in connections:
                if start_node in nodes and end_node in nodes:
                    start_idx = nodes.index(start_node)
                    end_idx = nodes.index(end_node)
                    
                    edge_x.extend([node_x[start_idx], node_x[end_idx], None])
                    edge_y.extend([node_y[start_idx], node_y[end_idx], None])
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False,
                name='Connections'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                ),
                hoverinfo='text',
                showlegend=False,
                name='Workflow Nodes'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'LangGraph Workflow: {analysis_type.title()} Analysis',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create workflow graph: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None