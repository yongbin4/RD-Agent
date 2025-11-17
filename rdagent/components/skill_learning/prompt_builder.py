"""Build prompts enhanced with skills and SOTA code."""

from typing import List, Dict, Optional
import logging

from rdagent.components.skill_learning.skill import Skill
from rdagent.components.experiment_learning.sota import SOTAModel

logger = logging.getLogger(__name__)


class SkillAwarePromptBuilder:
    """Build code generation prompts enhanced with skills and SOTA code."""

    def enhance_prompt(
        self,
        original_prompt: str,
        relevant_skills: List[Skill] = None,
        sota_code: Optional[Dict[str, str]] = None,
        max_skills: int = 3
    ) -> str:
        """
        Enhance the original prompt with skills and SOTA code.

        Args:
            original_prompt: The base prompt
            relevant_skills: List of relevant skills
            sota_code: Dict of SOTA code files (filename -> code)
            max_skills: Maximum number of skills to include

        Returns:
            Enhanced prompt string
        """
        if not relevant_skills and not sota_code:
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

        # Combine with original prompt
        if enhancement:
            enhanced = f"""{original_prompt}

{enhancement}

Use the above SOTA code and/or relevant patterns as reference when generating your solution.
Focus on building upon what worked before while improving and innovating.
"""
            logger.debug(f"Enhanced prompt with {len(relevant_skills or [])} skills and {'SOTA code' if sota_code else 'no SOTA'}")
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

    def build_task_prompt_with_context(
        self,
        task_description: str,
        relevant_skills: List[Skill] = None,
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

        return prompt
