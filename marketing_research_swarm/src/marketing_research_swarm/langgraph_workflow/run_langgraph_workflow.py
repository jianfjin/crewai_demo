#!/usr/bin/env python3
"""
LangGraph Workflow Runner

This script provides a command-line interface to run the LangGraph-based
marketing research workflow, replacing the CrewAI implementation.
"""

import argparse
import logging
import json
import sys
import os
from typing import List, Dict, Any
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from marketing_research_swarm.langgraph_workflow.workflow import MarketingResearchWorkflow
from marketing_research_swarm.langgraph_workflow.dashboard_integration import DashboardWorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load workflow configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_file}: {e}")
        return {}


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        "selected_agents": [
            "market_research_analyst",
            "data_analyst",
            "content_strategist",
            "campaign_optimizer"
        ],
        "workflow_params": {
            "target_audience": "health-conscious millennials and premium beverage consumers",
            "campaign_type": "multi-channel global marketing campaign",
            "budget": 250000,
            "duration": "12 months",
            "analysis_focus": "global beverage market performance and brand optimization",
            "business_objective": "Optimize beverage portfolio performance across global markets",
            "market_segments": ["North America", "Europe", "Asia Pacific"],
            "product_categories": ["Cola", "Juice", "Energy", "Sports"],
            "brands": ["Coca-Cola", "Pepsi", "Red Bull"],
            "campaign_goals": [
                "Optimize brand portfolio performance across global markets",
                "Identify high-margin opportunities by category and region",
                "Develop pricing strategies based on profitability analysis"
            ],
            "data_file_path": "data/beverage_sales_data.csv"
        }
    }


def run_workflow_cli(
    selected_agents: List[str],
    workflow_params: Dict[str, Any],
    output_file: str = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the workflow from command line interface.
    
    Args:
        selected_agents: List of agent names to include
        workflow_params: Parameters for the workflow
        output_file: Optional file to save results
        verbose: Enable verbose logging
        
    Returns:
        Workflow execution results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Initializing LangGraph Marketing Research Workflow")
    logger.info(f"Selected agents: {', '.join(selected_agents)}")
    
    try:
        # Initialize workflow
        workflow = MarketingResearchWorkflow()
        
        # Execute workflow
        logger.info("Starting workflow execution...")
        start_time = datetime.now()
        
        result = workflow.execute_workflow(
            selected_agents=selected_agents,
            **workflow_params
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Workflow execution completed in {execution_time:.2f} seconds")
        
        if result["success"]:
            logger.info(f"Workflow {result['workflow_id']} completed successfully")
            
            # Print summary
            summary = result.get("summary", {})
            print("\n" + "="*60)
            print("WORKFLOW EXECUTION SUMMARY")
            print("="*60)
            print(f"Workflow ID: {result['workflow_id']}")
            print(f"Status: {result['status']}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Total Agents: {summary.get('total_agents', 0)}")
            print(f"Completed Agents: {summary.get('completed_agents', 0)}")
            print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
            print("="*60)
            
            # Print agent results summary
            agent_results = result.get("agent_results", {})
            if agent_results:
                print("\nAGENT RESULTS SUMMARY:")
                print("-" * 40)
                for agent_name, agent_result in agent_results.items():
                    print(f"\n🤖 {agent_name.replace('_', ' ').title()}:")
                    if isinstance(agent_result, dict):
                        if 'analysis' in agent_result:
                            # Print first 200 characters of analysis
                            analysis = agent_result['analysis']
                            if len(analysis) > 200:
                                print(f"   {analysis[:200]}...")
                            else:
                                print(f"   {analysis}")
                        
                        if 'tool_results' in agent_result:
                            tool_count = len(agent_result['tool_results'])
                            print(f"   📊 {tool_count} tool result(s) generated")
            
            # Save results to file if specified
            if output_file:
                try:
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    logger.info(f"Results saved to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save results to {output_file}: {e}")
            
        else:
            logger.error(f"Workflow execution failed: {result.get('error', 'Unknown error')}")
            print(f"\n❌ Workflow execution failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow execution failed with exception: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run LangGraph Marketing Research Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_langgraph_workflow.py --sample

  # Run with specific agents
  python run_langgraph_workflow.py --agents market_research_analyst data_analyst

  # Run with configuration file
  python run_langgraph_workflow.py --config config.json

  # Run with custom parameters
  python run_langgraph_workflow.py --agents market_research_analyst data_analyst \\
    --target-audience "young professionals" --budget 100000
        """
    )
    
    # Agent selection
    parser.add_argument(
        '--agents',
        nargs='+',
        help='List of agents to include in the workflow',
        choices=[
            'market_research_analyst',
            'competitive_analyst',
            'data_analyst',
            'content_strategist',
            'creative_copywriter',
            'campaign_optimizer',
            'brand_performance_specialist',
            'forecasting_specialist'
        ]
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        help='JSON configuration file path'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample configuration for testing'
    )
    
    # Workflow parameters
    parser.add_argument(
        '--target-audience',
        default='health-conscious millennials and premium beverage consumers',
        help='Target audience for the campaign'
    )
    
    parser.add_argument(
        '--campaign-type',
        default='multi-channel global marketing campaign',
        help='Type of marketing campaign'
    )
    
    parser.add_argument(
        '--budget',
        type=float,
        default=250000,
        help='Campaign budget in dollars'
    )
    
    parser.add_argument(
        '--duration',
        default='12 months',
        help='Campaign duration'
    )
    
    parser.add_argument(
        '--analysis-focus',
        default='global beverage market performance and brand optimization',
        help='Focus area for the analysis'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        help='Output file to save results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Determine configuration source
    if args.config:
        config = load_config_from_file(args.config)
        if not config:
            print(f"❌ Failed to load configuration from {args.config}")
            sys.exit(1)
        
        selected_agents = config.get('selected_agents', [])
        workflow_params = config.get('workflow_params', {})
        
    elif args.sample:
        config = create_sample_config()
        selected_agents = config['selected_agents']
        workflow_params = config['workflow_params']
        
    else:
        # Use command line arguments
        selected_agents = args.agents or ['market_research_analyst', 'data_analyst']
        workflow_params = {
            'target_audience': args.target_audience,
            'campaign_type': args.campaign_type,
            'budget': args.budget,
            'duration': args.duration,
            'analysis_focus': args.analysis_focus
        }
    
    # Validate agents
    if not selected_agents:
        print("❌ No agents specified. Use --agents, --config, or --sample")
        sys.exit(1)
    
    # Run workflow
    result = run_workflow_cli(
        selected_agents=selected_agents,
        workflow_params=workflow_params,
        output_file=args.output,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()