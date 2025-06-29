"""
Marketing Research Swarm - AI-Powered Marketing Analytics

A comprehensive marketing research analysis platform powered by CrewAI's multi-agent swarm technology.
"""

__version__ = "0.1.0"
__author__ = "Marketing Research Swarm Team"
__email__ = "team@marketingresearchswarm.com"

from .crew import MarketingResearchCrew
from .main import run

__all__ = ["MarketingResearchCrew", "run"]