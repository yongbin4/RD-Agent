"""Build prompts enhanced with skills and SOTA code."""

from typing import List, Dict, Optional
import logging

from rdagent.components.skill_learning.skill import Skill
from rdagent.components.skill_learning.debug_skill import DebugSkill
from rdagent.components.experiment_learning.sota import SOTAModel

logger = logging.getLogger(__name__)


class SkillAwarePromptBuilder:
    """Build code generation prompts enhanced with skills and SOTA code."""

    def enhance_prompt(
        self,
        original_prompt: str,
        relevant_skills: List[Skill] = None,
        debug_skills: List[DebugSkill] = None,
        sota_code: Optional[Dict[str, str]] = None,
        max_skills: int = 3,
        max_debug_skills: int = 2
    ) -> str:
        """
        Enhance the original prompt with skills, debug skills, and SOTA code.

        Args:
            original_prompt: The base prompt
            relevant_skills: List of relevant success-based skills
            debug_skills: List of relevant debugging/problem-solving skills
            sota_code: Dict of SOTA code files (filename -> code)
            max_skills: Maximum number of success skills to include
            max_debug_skills: Maximum number of debug skills to include

        Returns:
            Enhanced prompt string
        """
        if not relevant_skills and not debug_skills and not sota_code:
            # Nothing to add
            return original_prompt

        # Build enhancement sections
        enhancement = ""

        # Add SOTA baseline section
        if sota_code:
            enhancement += self._build_sota_section(sota_code)

        # Add skills section
        if relevant_skills:
            enhancement += self._build_skills_section(relevant_skills[:max_skills])

        # Add debug skills section (common pitfalls)
        if debug_skills:
            enhancement += self._build_debug_skills_section(debug_skills[:max_debug_skills])

        # Combine with original prompt
        if enhancement:
            guidance = "Use the above SOTA code and/or relevant patterns as reference when generating your solution."
            if debug_skills:
                guidance += "\nPay special attention to the common pitfalls section to avoid known issues."
            guidance += "\nFocus on building upon what worked before while improving and innovating."

            enhanced = f"""{original_prompt}

{enhancement}

{guidance}
"""
            logger.debug(f"Enhanced prompt with {len(relevant_skills or [])} skills, {len(debug_skills or [])} debug skills, and {'SOTA code' if sota_code else 'no SOTA'}")
            return enhanced
        else:
            return original_prompt

    def _build_sota_section(self, sota_code: Dict[str, str]) -> str:
        """Build the SOTA baseline section."""
        section = """
## 🏆 Current Best Solution (SOTA Baseline)

The following code represents the current state-of-the-art solution for this competition.
Your goal is to improve upon this baseline.

"""
        for filename, code in list(sota_code.items())[:3]:  # Limit to 3 files to avoid token overflow
            # Truncate very long code
            if len(code) > 1500:
                code = code[:1500] + "\n# ... (truncated for brevity)"

            section += f"""### File: {filename}
```python
{code}
```

"""
        return section

    def _build_skills_section(self, skills: List[Skill]) -> str:
        """Build the relevant skills section."""
        section = """
## 📚 Relevant Patterns & Techniques

Based on successful experiments from past competitions, here are proven patterns that may be applicable:

"""
        for i, skill in enumerate(skills, 1):
            # Get the first example
            example_code = skill.code_pattern
            if skill.examples:
                example_code = skill.examples[0].code[:800]  # Limit length

            section += f"""### Pattern {i}: {skill.name.replace('_', ' ').title()}

**Description**: {skill.description}

**When to use**: {', '.join(skill.applicable_contexts)}

**Success rate**: {skill.success_rate():.1%} across {skill.attempt_count} experiments

**Example**:
```python
{example_code}
```

"""
        return section

    def _build_debug_skills_section(self, debug_skills: List[DebugSkill]) -> str:
        """Build the debug skills (common pitfalls) section."""
        section = """
## ⚠️ Common Pitfalls to Avoid

Based on past failures and their solutions, be aware of these common issues:

"""
        for i, skill in enumerate(debug_skills, 1):
            severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}.get(skill.severity, "⚠️")

            section += f"""### {severity_emoji} Pitfall {i}: {skill.name.replace('_', ' ').title()}

**Symptom**: {skill.symptom[:300]}

**Root Cause**: {skill.root_cause[:300]}

**How to Avoid**:
- {skill.description[:400]}

**Failed Approach** (Don't do this):
```python
{skill.failed_approach[:600]}
```

**Correct Approach**:
```python
{skill.solution[:600]}
```

**Fix Success Rate**: {skill.fix_success_rate():.1%} across {skill.detection_count} encounters

---

"""
        return section

    def build_task_prompt_with_context(
        self,
        task_description: str,
        relevant_skills: List[Skill] = None,
        debug_skills: List[DebugSkill] = None,
        previous_attempt_code: Optional[str] = None,
        previous_error: Optional[str] = None
    ) -> str:
        """
        Build a task-specific prompt with relevant context.

        This is useful for iterative refinement where we have a previous attempt.
        """
        prompt = f"""## Task
{task_description}

"""

        # Add previous attempt context
        if previous_attempt_code and previous_error:
            prompt += f"""## Previous Attempt (Failed)

The following code was tried but encountered an error:

```python
{previous_attempt_code[:1000]}
```

**Error**:
```
{previous_error[:500]}
```

Please fix the error and improve the implementation.

"""

        # Add relevant skills
        if relevant_skills:
            prompt += self._build_skills_section(relevant_skills)
            prompt += "\nConsider applying these patterns to avoid common pitfalls and improve your solution.\n"

        # Add debug skills (especially relevant when there was an error)
        if debug_skills:
            prompt += self._build_debug_skills_section(debug_skills)
            prompt += "\nCarefully review these common pitfalls to avoid repeating known mistakes.\n"

        return prompt
