"""Experiment learning components for RD-Agent.

This module provides functionality for:
- Storing state-of-the-art (SOTA) models from competitions
- Resuming experiments from previous best solutions
- Managing competition state across runs
"""

from rdagent.components.experiment_learning.sota import SOTAModel
from rdagent.components.experiment_learning.initializer import (
    initialize_trace_from_sota,
    should_resume_from_sota,
)

__all__ = [
    "SOTAModel",
    "initialize_trace_from_sota",
    "should_resume_from_sota",
]
