"""Initialize competition state and trace from SOTA models."""

from typing import Tuple
import logging

from rdagent.core.proposal import Trace, ExperimentFeedback
from rdagent.components.experiment_learning.sota import SOTAModel
from rdagent.components.skill_learning.global_kb import GlobalKnowledgeBase

logger = logging.getLogger(__name__)


def initialize_trace_from_sota(
    competition_name: str,
    global_kb: GlobalKnowledgeBase,
    base_trace=None,
    top_k: int = 3
) -> Trace:
    """
    Initialize a trace with SOTA experiments.

    If SOTA models exist for the competition, they are loaded and added to the trace
    as successful experiments. This allows the system to resume from the best known
    solutions.

    Args:
        competition_name: Name of the competition
        global_kb: Global knowledge base
        base_trace: Optional base trace to extend (if None, creates new one)
        top_k: Number of SOTA models to load

    Returns:
        Trace initialized with SOTA experiments (or empty if no SOTA exists)
    """
    # Get SOTA models
    sota_models = global_kb.get_sota(competition_name, top_k=top_k)

    if not sota_models:
        logger.info(f"🆕 No SOTA found for {competition_name}, starting fresh")
        if base_trace:
            return base_trace
        # Return will be handled by caller to create appropriate trace type
        return None

    # Initialize or extend trace
    trace = base_trace

    logger.info(f"🔄 Resuming from SOTA for {competition_name}")
    logger.info(f"📊 Loading top {len(sota_models)} SOTA models:")

    # Add SOTA experiments to trace
    for sota in sota_models:
        try:
            # Convert SOTA to experiment
            exp = sota.to_experiment()

            # Create feedback
            feedback = ExperimentFeedback(
                reason=f"SOTA baseline (rank {sota.rank})",
                decision=True,  # Mark as successful
                score=sota.score if hasattr(sota, 'score') else None,
            )

            # Add to trace if trace supports it
            if trace and hasattr(trace, 'hist'):
                trace.hist.append((exp, feedback))

            logger.info(f"  ✅ Rank {sota.rank}: Score {sota.score:.6f}")

        except Exception as e:
            logger.error(f"Error converting SOTA rank {sota.rank} to experiment: {e}")

    if trace and hasattr(trace, 'hist'):
        logger.info(f"🎯 Initialized trace with {len(sota_models)} SOTA experiments")

    return trace


def should_resume_from_sota(competition_name: str, global_kb: GlobalKnowledgeBase) -> bool:
    """
    Check if we should resume from SOTA for a competition.

    Args:
        competition_name: Name of the competition
        global_kb: Global knowledge base

    Returns:
        True if SOTA exists and we should resume, False otherwise
    """
    sota_models = global_kb.get_sota(competition_name, top_k=1)
    return len(sota_models) > 0
