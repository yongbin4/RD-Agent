from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from rdagent.core.developer import Developer
from rdagent.core.evolving_agent import EvolvingStrategy, RAGEvaluator, RAGEvoAgent
from rdagent.core.evolving_framework import EvoStep
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import RD_Agent_TIMER_wrapper

if TYPE_CHECKING:
    from rdagent.components.skill_learning.debug_extractor import DebugSkillExtractor
    from rdagent.components.skill_learning.global_kb import GlobalKnowledgeBase


class CoSTEER(Developer[Experiment]):
    def __init__(
        self,
        settings: CoSTEERSettings,
        eva: RAGEvaluator,
        es: EvolvingStrategy,
        *args,
        evolving_version: int = 2,
        with_knowledge: bool = True,
        knowledge_self_gen: bool = True,
        max_loop: int | None = None,
        global_kb: "GlobalKnowledgeBase | None" = None,
        debug_skill_extractor: "DebugSkillExtractor | None" = None,
        competition_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings

        self.max_loop = settings.max_loop if max_loop is None else max_loop
        self.knowledge_base_path = (
            Path(settings.knowledge_base_path) if settings.knowledge_base_path is not None else None
        )
        self.new_knowledge_base_path = (
            Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path is not None else None
        )

        self.with_knowledge = with_knowledge
        self.knowledge_self_gen = knowledge_self_gen
        self.evolving_strategy = es
        self.evaluator = eva
        self.evolving_version = evolving_version

        # Global knowledge base for debug skill extraction
        self.global_kb = global_kb
        self.debug_skill_extractor = debug_skill_extractor
        self.competition_name = competition_name

        # init rag method
        self.rag = (
            CoSTEERRAGStrategyV2(
                settings=settings,
                former_knowledge_base_path=self.knowledge_base_path,
                dump_knowledge_base_path=self.new_knowledge_base_path,
                evolving_version=self.evolving_version,
            )
            if self.evolving_version == 2
            else CoSTEERRAGStrategyV1(
                settings=settings,
                former_knowledge_base_path=self.knowledge_base_path,
                dump_knowledge_base_path=self.new_knowledge_base_path,
                evolving_version=self.evolving_version,
            )
        )

    def get_develop_max_seconds(self) -> int | None:
        """
        Get the maximum seconds for the develop task.
        Sub classes might override this method to provide a different value.
        """
        return None

    def _get_last_fb(self) -> CoSTEERMultiFeedback:
        fb = self.evolve_agent.evolving_trace[-1].feedback
        assert fb is not None, "feedback is None"
        assert isinstance(fb, CoSTEERMultiFeedback), "feedback must be of type CoSTEERMultiFeedback"
        return fb

    def should_use_new_evo(self, base_fb: CoSTEERMultiFeedback | None, new_fb: CoSTEERMultiFeedback) -> bool:
        """
        Compare new feedback with the fallback feedback.

        Returns:
            bool: True if the new feedback better and False if the new feedback is worse or invalid.
        """
        if new_fb is not None and new_fb.is_acceptable():
            return True
        return False

    def develop(self, exp: Experiment) -> Experiment:

        # init intermediate items
        max_seconds = self.get_develop_max_seconds()
        evo_exp = EvolvingItem.from_experiment(exp)

        self.evolve_agent = RAGEvoAgent[EvolvingItem](
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            with_feedback=True,
            knowledge_self_gen=self.knowledge_self_gen,
            enable_filelock=self.settings.enable_filelock,
            filelock_path=self.settings.filelock_path,
        )

        # Evolving the solution
        start_datetime = datetime.now()
        fallback_evo_exp = None
        fallback_evo_fb = None
        reached_max_seconds = False

        evo_fb = None
        for evo_exp in self.evolve_agent.multistep_evolve(evo_exp, self.evaluator):
            assert isinstance(evo_exp, Experiment)  # multiple inheritance
            evo_fb = self._get_last_fb()
            update_fallback = self.should_use_new_evo(
                base_fb=fallback_evo_fb,
                new_fb=evo_fb,
            )
            if update_fallback:
                fallback_evo_exp = deepcopy(evo_exp)
                fallback_evo_fb = deepcopy(evo_fb)
                fallback_evo_exp.create_ws_ckp()  # NOTE: creating checkpoints for saving files in the workspace to prevent inplace mutation.

            logger.log_object(evo_exp.sub_workspace_list, tag="evolving code")
            for sw in evo_exp.sub_workspace_list:
                logger.info(f"evolving workspace: {sw}")
            if max_seconds is not None and (datetime.now() - start_datetime).total_seconds() > max_seconds:
                logger.info(f"Reached max time limit {max_seconds} seconds, stop evolving")
                reached_max_seconds = True
                break
            if RD_Agent_TIMER_wrapper.timer.started and RD_Agent_TIMER_wrapper.timer.is_timeout():
                logger.info("Global timer is timeout, stop evolving")
                break

        try:
            # Fallback is required because we might not choose the last acceptable evo to submit.
            if fallback_evo_exp is not None:
                logger.info("Fallback to the fallback solution.")
                evo_exp = fallback_evo_exp
                evo_exp.recover_ws_ckp()
                evo_fb = fallback_evo_fb
            assert evo_fb is not None  # multistep_evolve should run at least once
            evo_exp = self._exp_postprocess_by_feedback(evo_exp, evo_fb)
        except CoderError as e:
            e.caused_by_timeout = reached_max_seconds
            raise e

        exp.sub_workspace_list = evo_exp.sub_workspace_list
        exp.experiment_workspace = evo_exp.experiment_workspace

        # Extract debug skills from the evolving trace (failure→success transitions)
        self._extract_debug_skills_from_trace()

        return exp

    def _extract_debug_skills_from_trace(self) -> None:
        """
        Extract debug skills from CoSTEER evolving trace.

        Analyzes the evolving_trace for failure→success transitions at the task level.
        When code fails and is subsequently fixed, this represents a valuable debug pattern.
        """
        logger.debug(f"_extract_debug_skills_from_trace called: global_kb={self.global_kb is not None}, extractor={self.debug_skill_extractor is not None}")

        if self.global_kb is None or self.debug_skill_extractor is None:
            logger.debug("Skipping debug skill extraction: global_kb or debug_skill_extractor is None")
            return

        if not hasattr(self, 'evolve_agent') or not hasattr(self.evolve_agent, 'evolving_trace'):
            logger.debug("Skipping debug skill extraction: no evolve_agent or evolving_trace")
            return

        evolving_trace = self.evolve_agent.evolving_trace
        if len(evolving_trace) < 2:
            logger.debug(f"Skipping debug skill extraction: only {len(evolving_trace)} steps (need >= 2)")
            return

        logger.info(f"🔍 Analyzing {len(evolving_trace)} evo steps for debug skill extraction...")

        extracted_count = 0
        transitions_found = 0
        for i in range(1, len(evolving_trace)):
            prev_step = evolving_trace[i - 1]
            curr_step = evolving_trace[i]

            prev_fb = prev_step.feedback
            curr_fb = curr_step.feedback

            if prev_fb is None or curr_fb is None:
                logger.debug(f"Step {i}: feedback is None (prev={prev_fb is not None}, curr={curr_fb is not None})")
                continue

            # For PipelineTask, feedback is a single item (not multi-feedback)
            # For component-based tasks, feedback is CoSTEERMultiFeedback
            if isinstance(curr_fb, CoSTEERMultiFeedback):
                # Check each sub-task for failure→success
                for task_idx in range(len(curr_fb)):
                    prev_task_fb = prev_fb[task_idx] if task_idx < len(prev_fb) else None
                    curr_task_fb = curr_fb[task_idx]

                    if prev_task_fb and curr_task_fb:
                        prev_failed = not prev_task_fb.is_acceptable()
                        curr_succeeded = curr_task_fb.is_acceptable()

                        logger.debug(f"Step {i}, task {task_idx}: prev_failed={prev_failed}, curr_succeeded={curr_succeeded}")

                        if prev_failed and curr_succeeded:
                            transitions_found += 1
                            logger.info(f"🎯 Found failure→success transition at step {i}, task {task_idx}")
                            debug_skill = self._extract_single_debug_skill(
                                prev_step, curr_step, task_idx
                            )
                            if debug_skill:
                                extracted_count += 1
            else:
                # Single task feedback (e.g., PipelineTask)
                prev_failed = not prev_fb.is_acceptable() if hasattr(prev_fb, 'is_acceptable') else False
                curr_succeeded = curr_fb.is_acceptable() if hasattr(curr_fb, 'is_acceptable') else False

                logger.debug(f"Step {i} (single): prev_failed={prev_failed}, curr_succeeded={curr_succeeded}, fb_type={type(curr_fb).__name__}")

                if prev_failed and curr_succeeded:
                    transitions_found += 1
                    logger.info(f"🎯 Found failure→success transition at step {i}")
                    debug_skill = self._extract_single_debug_skill(
                        prev_step, curr_step, task_idx=0
                    )
                    if debug_skill:
                        extracted_count += 1

        logger.info(f"📊 Debug skill extraction summary: {transitions_found} transitions found, {extracted_count} skills extracted")
        if extracted_count > 0:
            logger.info(f"🔧 Extracted {extracted_count} debug skill(s) from CoSTEER evolving trace")

    def _extract_single_debug_skill(
        self,
        prev_step: EvoStep,
        curr_step: EvoStep,
        task_idx: int
    ):
        """
        Extract a single debug skill from a failure→success transition.

        Args:
            prev_step: The step where code failed
            curr_step: The step where code succeeded
            task_idx: Index of the sub-task (for multi-task scenarios)

        Returns:
            DebugSkill if successfully extracted, None otherwise
        """
        try:
            # Get the code from both steps
            prev_evo = prev_step.evolvable_subjects
            curr_evo = curr_step.evolvable_subjects

            # Extract code from workspaces
            prev_code = self._get_workspace_code(prev_evo, task_idx)
            curr_code = self._get_workspace_code(curr_evo, task_idx)

            if not prev_code or not curr_code:
                return None

            # Get feedback info
            prev_fb = prev_step.feedback
            curr_fb = curr_step.feedback

            # Build context for extraction
            if isinstance(prev_fb, CoSTEERMultiFeedback):
                prev_task_fb = prev_fb[task_idx]
                curr_task_fb = curr_fb[task_idx]
            else:
                prev_task_fb = prev_fb
                curr_task_fb = curr_fb

            # Create a minimal experiment-like object for the extractor
            class MinimalExp:
                def __init__(self, code_dict):
                    self.experiment_workspace = type('obj', (object,), {'file_dict': code_dict})()
                    self.hypothesis = None

            class MinimalFeedback:
                def __init__(self, fb, succeeded: bool):
                    self.observations = str(fb) if fb else ""
                    self.score = None
                    self.decision = succeeded

            failed_exp = MinimalExp(prev_code)
            success_exp = MinimalExp(curr_code)
            failed_feedback = MinimalFeedback(prev_task_fb, False)
            success_feedback = MinimalFeedback(curr_task_fb, True)

            # Extract the debug skill
            debug_skill = self.debug_skill_extractor.extract_from_transition(
                failed_experiment=failed_exp,
                success_experiment=success_exp,
                failed_feedback=failed_feedback,
                success_feedback=success_feedback,
                competition_context=self.competition_name,
            )

            if debug_skill:
                logger.info(f"🐛 Extracted debug skill: {debug_skill.name}")
                self.global_kb.add_or_update_debug_skill(debug_skill)
                return debug_skill

        except Exception as e:
            logger.warning(f"Error extracting debug skill: {e}")

        return None

    def _get_workspace_code(self, evo, task_idx: int) -> dict:
        """Extract code dictionary from evolving item workspace."""
        try:
            if hasattr(evo, 'sub_workspace_list') and evo.sub_workspace_list:
                if task_idx < len(evo.sub_workspace_list):
                    ws = evo.sub_workspace_list[task_idx]
                    if hasattr(ws, 'file_dict'):
                        return ws.file_dict
            if hasattr(evo, 'experiment_workspace'):
                ws = evo.experiment_workspace
                if hasattr(ws, 'file_dict'):
                    return ws.file_dict
        except Exception:
            pass
        return {}

    def _exp_postprocess_by_feedback(self, evo: Experiment, feedback: CoSTEERMultiFeedback) -> Experiment:
        """
        Responsibility:
        - Raise Error if it failed to handle the develop task
        -
        """
        assert isinstance(evo, Experiment)
        assert isinstance(feedback, CoSTEERMultiFeedback)
        assert len(evo.sub_workspace_list) == len(feedback)

        # FIXME: when whould the feedback be None?
        failed_feedbacks = [
            f"- feedback{index + 1:02d}:\n  - execution: {f.execution}\n  - return_checking: {f.return_checking}\n  - code: {f.code}"
            for index, f in enumerate(feedback)
            if f is not None and not f.is_acceptable()
        ]

        if len(failed_feedbacks) == len(feedback):
            feedback_summary = "\n".join(failed_feedbacks)
            raise CoderError(f"All tasks are failed:\n{feedback_summary}")

        return evo
