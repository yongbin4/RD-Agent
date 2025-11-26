import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.share.doc import DocDev
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError, PolicyError, RunnerError
from rdagent.core.proposal import ExperimentFeedback, ExpGen
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.base import DataScienceScen
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSKnowledgeBase
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import DSProposalV2ExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.trace_scheduler import (
    MCTSScheduler,
)
from rdagent.utils.workflow.misc import wait_retry


def clean_workspace(workspace_root: Path) -> None:
    """
    Clean the workspace folder and only keep the essential files to save more space.
    workspace_root might contain a file in parallel with the folders, we should directly remove it.

    # remove all files and folders in the workspace except for .py, .md, and .csv files to avoid large workspace dump
    """
    if workspace_root.is_file():
        workspace_root.unlink()
    else:
        for file_and_folder in workspace_root.iterdir():
            if file_and_folder.is_dir():
                if file_and_folder.is_symlink():
                    file_and_folder.unlink()
                else:
                    shutil.rmtree(file_and_folder)
            elif file_and_folder.is_file() and file_and_folder.suffix not in [".py", ".md", ".csv"]:
                file_and_folder.unlink()


@wait_retry()
def backup_folder(path: str | Path) -> Path:
    path = Path(path)
    workspace_bak_path = path.with_name(path.name + ".bak")
    if workspace_bak_path.exists():
        shutil.rmtree(workspace_bak_path)

    try:
        # `cp` may raise error if the workspace is beiing modified.
        # rsync is more robust choice, but it is not installed in some docker images.
        # use shutil.copytree(..., symlinks=True) should be more elegant, but it has more changes to raise error.
        subprocess.run(
            ["cp", "-r", "-P", str(path), str(workspace_bak_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error copying {path} to {workspace_bak_path}: {e}")
        logger.error(f"Stdout: {e.stdout.decode() if e.stdout else ''}")
        logger.error(f"Stderr: {e.stderr.decode() if e.stderr else ''}")
        raise
    return workspace_bak_path


class DataScienceRDLoop(RDLoop):
    # NOTE: we move the DataScienceRDLoop here to be easier to be imported
    skip_loop_error = (CoderError, RunnerError)
    withdraw_loop_error = (PolicyError,)

    # when using more advanced proposals(merged, parallel, etc.), we provide a default exp_gen for convinience.
    default_exp_gen: type[ExpGen] = DSProposalV2ExpGen

    def __init__(self, PROP_SETTING: BasePropSetting):
        logger.log_object(PROP_SETTING.competition, tag="competition")
        scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")

        # 1) task generation from scratch
        # self.scratch_gen: tuple[HypothesisGen, Hypothesis2Experiment] = DummyHypothesisGen(scen),

        # 2) task generation from a complete solution
        # self.exp_gen: ExpGen = import_class(PROP_SETTING.exp_gen)(scen)

        self.ckp_selector = import_class(PROP_SETTING.selector_name)()
        self.sota_exp_selector = import_class(PROP_SETTING.sota_exp_selector_name)()
        self.exp_gen: ExpGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

        self.interactor = import_class(PROP_SETTING.interactor)(scen)

        # coders
        self.data_loader_coder = DataLoaderCoSTEER(scen)
        self.feature_coder = FeatureCoSTEER(scen)
        self.model_coder = ModelCoSTEER(scen)
        self.ensemble_coder = EnsembleCoSTEER(scen)
        self.workflow_coder = WorkflowCoSTEER(scen)

        self.pipeline_coder = PipelineCoSTEER(scen)

        self.runner = DSCoSTEERRunner(scen)

        # Initialize global knowledge base (skill learning - always enabled)
        from rdagent.components.skill_learning.global_kb import GlobalKnowledgeBase
        from rdagent.components.skill_learning.extractor import SkillExtractor
        from rdagent.components.skill_learning.debug_extractor import DebugSkillExtractor

        self.competition_name = PROP_SETTING.competition
        self.global_kb = GlobalKnowledgeBase()
        self.skill_extractor = SkillExtractor()
        self.debug_skill_extractor = DebugSkillExtractor()

        # Track recent experiments for failure pattern detection
        self.experiment_history = []  # List of (experiment, feedback) tuples

        # Load all skills on startup
        self.global_kb.load_all_skills()
        self.global_kb.load_all_debug_skills()
        stats = self.global_kb.get_statistics()
        logger.info(f"📚 Global Knowledge Base initialized:")
        logger.info(f"  - {stats['total_skills']} skills loaded")
        logger.info(f"  - {stats['total_debug_skills']} debug skills loaded")
        logger.info(f"  - {stats['total_competitions']} competitions in history")
        logger.info(f"  - {stats['average_skill_success_rate']:.1%} average skill success rate")
        if stats['total_debug_skills'] > 0:
            logger.info(f"  - {stats['average_debug_skill_fix_rate']:.1%} average debug skill fix rate")

        # Pass global KB to all coders
        for coder in [self.data_loader_coder, self.feature_coder, self.model_coder,
                      self.ensemble_coder, self.workflow_coder, self.pipeline_coder]:
            coder.global_kb = self.global_kb
            coder.competition_name = self.competition_name
        if DS_RD_SETTING.enable_doc_dev:
            self.docdev = DocDev(scen)

        if DS_RD_SETTING.enable_knowledge_base and DS_RD_SETTING.knowledge_base_version == "v1":
            knowledge_base = DSKnowledgeBase(
                path=DS_RD_SETTING.knowledge_base_path, idea_pool_json_path=DS_RD_SETTING.idea_pool_json_path
            )
            self.trace = DSTrace(scen=scen, knowledge_base=knowledge_base)
        else:
            self.trace = DSTrace(scen=scen)

        # Auto-resume from SOTA if available
        from rdagent.components.experiment_learning.initializer import initialize_trace_from_sota, should_resume_from_sota

        if should_resume_from_sota(self.competition_name, self.global_kb):
            # Initialize trace with SOTA experiments
            self.trace = initialize_trace_from_sota(
                competition_name=self.competition_name,
                global_kb=self.global_kb,
                base_trace=self.trace,
                top_k=3
            )
            logger.info(f"🚀 Competition will build upon existing SOTA")

        self.summarizer = import_class(PROP_SETTING.summarizer)(scen=scen, **PROP_SETTING.summarizer_init_kwargs)

        super(RDLoop, self).__init__()

    def _has_valid_score(self, exp, feedback) -> bool:
        """
        Check if experiment completed successfully with a valid score.

        Returns True if the experiment produced a valid numeric score,
        regardless of whether it beat SOTA.
        """
        # Try to get score from feedback first
        score_raw = getattr(feedback, 'score', None)

        # Fallback: extract from exp.result
        if score_raw is None and hasattr(exp, 'result') and exp.result is not None:
            try:
                import pandas as pd
                df = pd.DataFrame(exp.result)
                if 'ensemble' in df.index:
                    score_raw = df.loc["ensemble"].iloc[0]
            except Exception:
                pass

        # Convert to float and validate
        if score_raw is not None:
            try:
                import numpy as np
                score = float(score_raw)
                return not (np.isnan(score) or np.isinf(score))
            except (ValueError, TypeError):
                pass

        return False

    def _detect_failure_patterns(self, current_exp, current_feedback) -> Optional[tuple[str, any, any]]:
        """
        Detect failure-to-success patterns for debug skill extraction.

        Returns:
            Tuple of (pattern_type, failed_exp/feedback, success_exp/feedback) if pattern found, None otherwise
            pattern_type: "consecutive", "error_fix", or "hypothesis_evolution"
        """
        # Pattern 1: Consecutive experiments (failed → succeeded)
        # Check if previous experiment failed and current one succeeded
        if len(self.experiment_history) >= 1:
            prev_exp, prev_feedback = self.experiment_history[-1]

            # Check if current succeeded and previous failed
            current_success = self._has_valid_score(current_exp, current_feedback)
            prev_failed = not self._has_valid_score(prev_exp, prev_feedback)

            if current_success and prev_failed:
                logger.info("🔍 Detected consecutive failure→success pattern")
                return ("consecutive", (prev_exp, prev_feedback), (current_exp, current_feedback))

        # Pattern 2: Error-based (feedback contains errors but experiment improved)
        # Check if current feedback mentions errors but decision is positive
        if hasattr(current_feedback, 'observations'):
            obs = str(current_feedback.observations).lower()
            has_error_mention = any(keyword in obs for keyword in ['error', 'exception', 'fix', 'solved', 'resolved'])
            current_success = self._has_valid_score(current_exp, current_feedback)

            if has_error_mention and current_success:
                logger.info("🔍 Detected error-fix pattern in feedback")
                return ("error_fix", current_feedback, (current_exp, current_feedback))

        # Pattern 3: Hypothesis evolution (similar hypothesis rejected before, now accepted)
        # Check if current hypothesis was previously rejected but now accepted
        if hasattr(current_exp, 'hypothesis') and self._has_valid_score(current_exp, current_feedback):
            current_hyp = str(current_exp.hypothesis).lower() if hasattr(current_exp.hypothesis, '__str__') else ""

            # Look for similar rejected hypotheses in history
            for prev_exp, prev_feedback in self.experiment_history[-5:]:  # Check last 5
                if hasattr(prev_exp, 'hypothesis') and not self._has_valid_score(prev_exp, prev_feedback):
                    prev_hyp = str(prev_exp.hypothesis).lower() if hasattr(prev_exp.hypothesis, '__str__') else ""

                    # Simple similarity check (can be improved with embeddings)
                    if prev_hyp and current_hyp and len(prev_hyp) > 20 and len(current_hyp) > 20:
                        # Check for common words (simple approach)
                        prev_words = set(prev_hyp.split())
                        curr_words = set(current_hyp.split())
                        overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))

                        if overlap > DS_RD_SETTING.debug_skill_hypothesis_overlap_threshold:
                            logger.info("🔍 Detected hypothesis evolution pattern")
                            return ("hypothesis_evolution", (prev_exp, prev_feedback), (current_exp, current_feedback))

        return None

    def _should_extract_skill_from_score(self, score: Optional[float]) -> tuple[bool, str]:
        """
        Determine if skill should be extracted based on relative score quality.

        For metrics where lower is better (like Log Loss):
        - Extract if score is in the top X% (lowest X%)
        - Bootstrap: always extract during first few experiments

        Args:
            score: The score of the current experiment (or None)

        Returns:
            (should_extract: bool, reason: str)
        """
        import numpy as np

        # Edge case: Score is None - skip
        if score is None:
            return False, "Score is None"

        # Get recent successful experiment scores from trace history
        recent_scores = []
        for exp, fb in self.trace.hist:
            if hasattr(fb, 'decision'):
                # Try to get score from feedback first
                exp_score_raw = getattr(fb, 'score', None)

                # Fallback: extract from exp.result if not in feedback
                if exp_score_raw is None and hasattr(exp, 'result') and exp.result is not None:
                    try:
                        import pandas as pd
                        df = pd.DataFrame(exp.result)
                        if 'ensemble' in df.index:
                            exp_score_raw = df.loc["ensemble"].iloc[0]
                    except Exception:
                        continue

                # Convert to float and validate
                if exp_score_raw is not None:
                    try:
                        exp_score = float(exp_score_raw)
                        if not np.isnan(exp_score) and not np.isinf(exp_score):
                            recent_scores.append(exp_score)
                    except (ValueError, TypeError):
                        continue

        # Bootstrap phase: not enough samples yet - always extract
        min_samples = DS_RD_SETTING.skill_extraction_min_samples
        if len(recent_scores) < min_samples:
            return True, f"Bootstrapping: only {len(recent_scores)} samples (need {min_samples})"

        # Calculate percentile cutoff for "lower is better" metrics
        percentile_threshold = DS_RD_SETTING.skill_extraction_percentile
        cutoff = np.percentile(recent_scores, percentile_threshold)

        # Lower scores are better, so extract if score <= cutoff
        should_extract = score <= cutoff

        reason = (
            f"Score {score:.4f} {'<=' if should_extract else '>'} "
            f"cutoff {cutoff:.4f} (top {percentile_threshold}% of {len(recent_scores)} samples)"
        )

        return should_extract, reason

    async def direct_exp_gen(self, prev_out: dict[str, Any]):

        # set the checkpoint to start from
        selection = self.ckp_selector.get_selection(self.trace)
        # set the current selection for the trace
        self.trace.set_current_selection(selection)

        # in parallel + multi-trace mode, the above global "trace.current_selection" will not be used
        # instead, we will use the "local_selection" attached to each exp to in async_gen().
        exp = await self.exp_gen.async_gen(self.trace, self)
        exp = self.interactor.interact(exp, self.trace)

        logger.log_object(exp)
        return exp

    def coding(self, prev_out: dict[str, Any]):
        exp = prev_out["direct_exp_gen"]
        for tasks in exp.pending_tasks_list:
            exp.sub_tasks = tasks
            with logger.tag(f"{exp.sub_tasks[0].__class__.__name__}"):
                if isinstance(exp.sub_tasks[0], DataLoaderTask):
                    exp = self.data_loader_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], FeatureTask):
                    exp = self.feature_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], ModelTask):
                    exp = self.model_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], EnsembleTask):
                    exp = self.ensemble_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], WorkflowTask):
                    exp = self.workflow_coder.develop(exp)
                elif isinstance(exp.sub_tasks[0], PipelineTask):
                    exp = self.pipeline_coder.develop(exp)
                else:
                    raise NotImplementedError(f"Unsupported component in DataScienceRDLoop: {exp.hypothesis.component}")
            exp.sub_tasks = []
        logger.log_object(exp)
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp: DSExperiment = prev_out["coding"]
        if exp.is_ready_to_run():
            new_exp = self.runner.develop(exp)
            logger.log_object(new_exp)
            exp = new_exp
        if DS_RD_SETTING.enable_doc_dev:
            self.docdev.develop(exp)
        return exp

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        """
        Assumption:
        - If we come to feedback phase, the previous development steps are successful.
        """
        exp: DSExperiment = prev_out["running"]

        # set the local selection to the trace after feedback
        if exp.local_selection is not None:
            self.trace.set_current_selection(exp.local_selection)

        if self.trace.next_incomplete_component() is None or DS_RD_SETTING.coder_on_whole_pipeline:
            # we have alreadly completed components in previous trace. So current loop is focusing on a new proposed idea.
            # So we need feedback for the proposal.
            feedback = self.summarizer.generate_feedback(exp, self.trace)
        else:
            # Otherwise, it is on drafting stage, don't need complicated feedbacks.
            feedback = ExperimentFeedback(
                reason=f"{exp.hypothesis.component} is completed.",
                decision=True,
            )
        logger.log_object(feedback)
        return feedback

    def record(self, prev_out: dict[str, Any]):

        exp: DSExperiment = None

        cur_loop_id = prev_out[self.LOOP_IDX_KEY]

        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is None:
            exp = prev_out["running"]

            # NOTE: we put below  operations on selections here, instead of out of the if-else block,
            # to fit the corner case that the trace will be reset

            # set the local selection to the trace as global selection, then set the DAG parent for the trace
            if exp.local_selection is not None:
                self.trace.set_current_selection(exp.local_selection)
            self.trace.sync_dag_parent_and_hist((exp, prev_out["feedback"]), cur_loop_id)

            # Skill learning: Extract skills from successful experiments
            feedback = prev_out["feedback"]
            if hasattr(self, 'global_kb') and hasattr(feedback, 'decision'):
                # Try to get score from feedback first (DSRunnerFeedback has score attribute)
                score_raw = getattr(feedback, 'score', None)

                # Fallback: extract from exp.result if not in feedback (for HypothesisFeedback)
                if score_raw is None and hasattr(exp, 'result') and exp.result is not None:
                    try:
                        import pandas as pd
                        df = pd.DataFrame(exp.result)
                        if 'ensemble' in df.index:
                            score_raw = df.loc["ensemble"].iloc[0]
                            logger.info(f"Extracted score from exp.result: {score_raw}")
                    except Exception as e:
                        logger.info(f"Could not extract score from exp.result: {e}")

                # Convert score to float
                score = None
                if score_raw is not None:
                    try:
                        score = float(score_raw)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert score to float: {score_raw}")
                        score = None

                # Check if we should extract skills based on relative score quality
                should_extract, reason = self._should_extract_skill_from_score(score)

                if should_extract:
                    logger.info(f"📊 Skill extraction triggered: {reason}")
                    try:
                        skills = self.skill_extractor.extract_from_experiment(
                            experiment=exp,
                            feedback=feedback,
                            competition_context=self.competition_name
                        )
                        for skill in skills:
                            self.global_kb.add_or_update_skill(skill)
                            logger.info(f"💡 Learned new skill: {skill.name}")
                    except Exception as e:
                        logger.error(f"Error extracting skills: {e}")
                else:
                    logger.info(f"⏭️  Skipping skill extraction: {reason}")

                # Update SOTA if this is better than current best
                # For metrics like Log Loss where lower is better, we compare appropriately
                if score is not None:
                    try:
                        current_best = self.global_kb.get_best_score(self.competition_name)
                        is_better = current_best is None or score < current_best  # Lower is better for Log Loss

                        if is_better:
                            from rdagent.components.experiment_learning.sota import SOTAModel
                            # Determine rank (lower score = better rank)
                            existing_sota = self.global_kb.get_sota(self.competition_name, top_k=3)
                            rank = len([s for s in existing_sota if s.score <= score]) + 1
                            rank = min(rank, 3)  # Keep only top 3

                            sota = SOTAModel.from_experiment(exp, feedback, rank=rank)
                            sota.competition = self.competition_name
                            self.global_kb.save_sota(self.competition_name, sota)
                            logger.info(f"🏆 New SOTA for {self.competition_name}: rank {rank}, score {score:.6f}")
                    except Exception as e:
                        logger.error(f"Error saving SOTA: {e}")

            # Debug skill learning: Extract problem-solving patterns from failures
            if DS_RD_SETTING.enable_debug_skill_extraction and hasattr(self, 'debug_skill_extractor') and hasattr(self, 'global_kb'):
                try:
                    # Detect failure patterns
                    pattern = self._detect_failure_patterns(exp, feedback)

                    if pattern:
                        pattern_type, failed_data, success_data = pattern
                        logger.info(f"🐛 Detected {pattern_type} pattern, extracting debug skill...")

                        debug_skill = None
                        if pattern_type == "consecutive":
                            # Extract from failure-success transition
                            failed_exp, failed_feedback = failed_data
                            success_exp, success_feedback = success_data
                            debug_skill = self.debug_skill_extractor.extract_from_transition(
                                failed_experiment=failed_exp,
                                success_experiment=success_exp,
                                failed_feedback=failed_feedback,
                                success_feedback=success_feedback,
                                competition_context=self.competition_name
                            )
                        elif pattern_type == "error_fix":
                            # Extract from error fix
                            debug_skill = self.debug_skill_extractor.extract_from_error_fix(
                                experiment=exp,
                                feedback=feedback,
                                competition_context=self.competition_name
                            )
                        elif pattern_type == "hypothesis_evolution":
                            # Extract from hypothesis evolution
                            failed_exp, failed_feedback = failed_data
                            success_exp, success_feedback = success_data
                            debug_skill = self.debug_skill_extractor.extract_from_transition(
                                failed_experiment=failed_exp,
                                success_experiment=success_exp,
                                failed_feedback=failed_feedback,
                                success_feedback=success_feedback,
                                competition_context=self.competition_name
                            )

                        if debug_skill:
                            self.global_kb.add_or_update_debug_skill(debug_skill)
                            logger.info(f"🔧 Learned debug skill: {debug_skill.name} (severity: {debug_skill.severity})")
                except Exception as e:
                    logger.error(f"Error extracting debug skill: {e}")

            # Add to experiment history (keep last N experiments based on config)
            if hasattr(self, 'experiment_history'):
                self.experiment_history.append((exp, feedback))
                max_history = DS_RD_SETTING.debug_skill_history_window
                if len(self.experiment_history) > max_history:
                    self.experiment_history.pop(0)

        else:
            exp: DSExperiment = prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"]
            # TODO: distinguish timeout error & other exception.
            if (
                isinstance(self.trace.scen, DataScienceScen)
                and DS_RD_SETTING.allow_longer_timeout
                and isinstance(e, CoderError)
                and e.caused_by_timeout
            ):
                logger.info(
                    f"Timeout error occurred: {e}. Increasing timeout for the current scenario from {self.trace.scen.timeout_increase_count} to {self.trace.scen.timeout_increase_count + 1}."
                )
                self.trace.scen.increase_timeout()

            # set the local selection to the trace as global selection, then set the DAG parent for the trace
            if exp.local_selection is not None:
                self.trace.set_current_selection(exp.local_selection)

            self.trace.sync_dag_parent_and_hist(
                (
                    exp,
                    ExperimentFeedback.from_exception(e),
                ),
                cur_loop_id,
            )
            # Value backpropagation is handled in async_gen before next() via observe_commits

            # Add failed experiment to history for debugging skill extraction
            if hasattr(self, 'experiment_history'):
                error_feedback = ExperimentFeedback.from_exception(e)
                self.experiment_history.append((exp, error_feedback))
                max_history = DS_RD_SETTING.debug_skill_history_window
                if len(self.experiment_history) > max_history:
                    self.experiment_history.pop(0)

            if self.trace.sota_experiment() is None:
                if DS_RD_SETTING.coder_on_whole_pipeline:
                    #  check if feedback is not generated
                    if len(self.trace.hist) >= DS_RD_SETTING.coding_fail_reanalyze_threshold:
                        recent_hist = self.trace.hist[-DS_RD_SETTING.coding_fail_reanalyze_threshold :]
                        if all(isinstance(fb.exception, (CoderError, RunnerError)) for _, fb in recent_hist):
                            new_scen = self.trace.scen
                            if hasattr(new_scen, "reanalyze_competition_description"):
                                logger.info(
                                    "Reanalyzing the competition description after three consecutive coding failures."
                                )
                                new_scen.reanalyze_competition_description()
                                self.trace.scen = new_scen
                            else:
                                logger.info("Can not reanalyze the competition description.")
                elif len(self.trace.hist) >= DS_RD_SETTING.consecutive_errors:
                    # if {in inital/drafting stage} and {tried enough times}
                    for _, fb in self.trace.hist[-DS_RD_SETTING.consecutive_errors :]:
                        if fb:
                            break  # any success will stop restarting.
                    else:  # otherwise restart it
                        logger.error("Consecutive errors reached the limit. Dumping trace.")
                        logger.log_object(self.trace, tag="trace before restart")
                        self.trace = DSTrace(scen=self.trace.scen, knowledge_base=self.trace.knowledge_base)
                        # Reset the trace; MCTS stats will be cleared via registered callback
                        self.exp_gen.reset()

        # set the SOTA experiment to submit
        sota_exp_to_submit = self.sota_exp_selector.get_sota_exp_to_submit(self.trace)
        self.trace.set_sota_exp_to_submit(sota_exp_to_submit)
        logger.log_object(sota_exp_to_submit, tag="sota_exp_to_submit")

        logger.log_object(self.trace, tag="trace")
        logger.log_object(self.trace.sota_experiment(search_type="all"), tag="SOTA experiment")

        if DS_RD_SETTING.enable_knowledge_base and DS_RD_SETTING.knowledge_base_version == "v1":
            logger.log_object(self.trace.knowledge_base, tag="knowledge_base")
            self.trace.knowledge_base.dump()

        if (
            DS_RD_SETTING.enable_log_archive
            and DS_RD_SETTING.log_archive_path is not None
            and Path(DS_RD_SETTING.log_archive_path).is_dir()
        ):
            start_archive_datetime = datetime.now()
            logger.info(f"Archiving log and workspace folder after loop {self.loop_idx}")
            mid_log_tar_path = (
                Path(
                    DS_RD_SETTING.log_archive_temp_path
                    if DS_RD_SETTING.log_archive_temp_path
                    else DS_RD_SETTING.log_archive_path
                )
                / "mid_log.tar"
            )
            mid_workspace_tar_path = (
                Path(
                    DS_RD_SETTING.log_archive_temp_path
                    if DS_RD_SETTING.log_archive_temp_path
                    else DS_RD_SETTING.log_archive_path
                )
                / "mid_workspace.tar"
            )
            log_back_path = backup_folder(Path().cwd() / "log")
            subprocess.run(["tar", "-cf", str(mid_log_tar_path), "-C", str(log_back_path), "."], check=True)

            # only clean current workspace without affecting other loops.
            for k in "direct_exp_gen", "coding", "running":
                if k in prev_out and prev_out[k] is not None:
                    assert isinstance(prev_out[k], DSExperiment)
                    clean_workspace(prev_out[k].experiment_workspace.workspace_path)

            # Backup the workspace (only necessary files are included)
            # - Step 1: Copy the workspace to a .bak package
            workspace_bak_path = backup_folder(RD_AGENT_SETTINGS.workspace_path)

            # - Step 2: Clean .bak package
            for bak_workspace in workspace_bak_path.iterdir():
                clean_workspace(bak_workspace)

            # - Step 3: Create tarball from the cleaned .bak workspace
            subprocess.run(["tar", "-cf", str(mid_workspace_tar_path), "-C", str(workspace_bak_path), "."], check=True)

            # - Step 4: Remove .bak package
            shutil.rmtree(workspace_bak_path)

            if DS_RD_SETTING.log_archive_temp_path is not None:
                shutil.move(mid_log_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_log.tar")
                mid_log_tar_path = Path(DS_RD_SETTING.log_archive_path) / "mid_log.tar"
                shutil.move(mid_workspace_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_workspace.tar")
                mid_workspace_tar_path = Path(DS_RD_SETTING.log_archive_path) / "mid_workspace.tar"
            shutil.copy(
                mid_log_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_log_bak.tar"
            )  # backup when upper code line is killed when running
            shutil.copy(
                mid_workspace_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_workspace_bak.tar"
            )  # backup when upper code line is killed when running
            self.timer.add_duration(datetime.now() - start_archive_datetime)

    def _check_exit_conditions_on_step(self, loop_id: Optional[int] = None, step_id: Optional[int] = None):
        if step_id not in [self.steps.index("running"), self.steps.index("feedback")]:
            # pass the check for running and feedbacks since they are very likely to be finished soon.
            super()._check_exit_conditions_on_step(loop_id=loop_id, step_id=step_id)

    @classmethod
    def load(
        cls,
        path: str | Path,
        checkout: bool | str | Path = False,
        replace_timer: bool = True,
    ) -> "LoopBase":
        session = super().load(path, checkout, replace_timer)
        logger.log_object(DS_RD_SETTING.competition, tag="competition")  # NOTE: necessary to make mle_summary work.
        if DS_RD_SETTING.enable_knowledge_base and DS_RD_SETTING.knowledge_base_version == "v1":
            session.trace.knowledge_base = DSKnowledgeBase(
                path=DS_RD_SETTING.knowledge_base_path, idea_pool_json_path=DS_RD_SETTING.idea_pool_json_path
            )
        return session

    def dump(self, path: str | Path) -> None:
        """
        Since knowledge_base is big and we don't want to dump it every time
        So we remove it from the trace before dumping and restore it after.
        """
        backup_knowledge_base = None
        if self.trace.knowledge_base is not None:
            backup_knowledge_base = self.trace.knowledge_base
            self.trace.knowledge_base = None
        super().dump(path)
        if backup_knowledge_base is not None:
            self.trace.knowledge_base = backup_knowledge_base
