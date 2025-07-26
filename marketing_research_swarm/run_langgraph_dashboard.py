#!/usr/bin/env python3
"""
LangGraph Dashboard Runner

Launch the optimized LangGraph marketing research dashboard with token optimization.
"""

import sys
import os
import subprocess
import logging

# Add the project root to the Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    
    print("🔍 Checking Dependencies...")
    
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "langgraph",
        "langchain_openai"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_langgraph_components():
    """Check if LangGraph components are available."""
    
    print("\n🔍 Checking LangGraph Components...")
    
    try:
        from marketing_research_swarm.langgraph_workflow.workflow import MarketingResearchWorkflow
        print("✅ Standard LangGraph workflow")
    except ImportError as e:
        print(f"❌ Standard workflow: {e}")
        return False
    
    try:
        from marketing_research_swarm.langgraph_workflow.optimized_workflow import OptimizedMarketingWorkflow
        print("✅ Optimized LangGraph workflow")
    except ImportError as e:
        print(f"❌ Optimized workflow: {e}")
        return False
    
    try:
        from marketing_research_swarm.optimization_manager import OptimizationManager
        print("✅ Optimization manager")
    except ImportError as e:
        print(f"❌ Optimization manager: {e}")
        return False
    
    return True


def setup_environment():
    """Setup environment variables and configuration."""
    
    print("\n🔧 Setting up Environment...")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return False
    else:
        print("✅ OPENAI_API_KEY configured")
    
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"✅ Created cache directory: {cache_dir}")
    else:
        print(f"✅ Cache directory exists: {cache_dir}")
    
    return True


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    
    print("\n🚀 Launching LangGraph Dashboard...")
    print("=" * 50)
    print("Dashboard Features:")
    print("• 🤖 LangGraph workflow orchestration")
    print("• ⚡ Token optimization (75-85% reduction)")
    print("• 💾 Smart caching system")
    print("• 📊 Real-time performance monitoring")
    print("• 🎯 Multiple optimization levels")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "langgraph_dashboard.py",
            "--server.port", "8502",
            "--server.address", "0.0.0.0",
            "--theme.base", "light"
        ]
        
        print(f"Starting dashboard on http://localhost:8502")
        print("Press Ctrl+C to stop the dashboard")
        print()
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    
    return True


def main():
    """Main function to run the dashboard."""
    
    print("🚀 LangGraph Marketing Research Dashboard")
    print("=" * 60)
    print("Optimized workflow with intelligent token reduction")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        return 1
    
    # Check LangGraph components
    if not check_langgraph_components():
        print("\n❌ LangGraph component check failed")
        print("Make sure the optimized workflow is properly implemented")
        return 1
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed")
        return 1
    
    print("\n✅ All checks passed!")
    print("\n" + "=" * 60)
    
    # Offer to run optimization test first
    run_test = input("🧪 Run optimization test first? (y/n): ").strip().lower()
    if run_test == 'y':
        print("\n🧪 Running optimization test...")
        try:
            from test_langgraph_optimization import test_langgraph_optimization
            test_langgraph_optimization()
            print("\n✅ Optimization test completed")
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
        
        input("\nPress Enter to continue to dashboard...")
    
    # Launch dashboard
    success = launch_dashboard()
    
    if success:
        print("\n✅ Dashboard session completed successfully")
        return 0
    else:
        print("\n❌ Dashboard launch failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)