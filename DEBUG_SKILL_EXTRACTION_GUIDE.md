# Debug Skill Extraction - Implementation Guide & Validation

## Overview

Debug skill extraction detects when code goes from "failed/crashed" to "working" and learns reusable patterns from these transitions.

**Two extraction points:**
1. **Loop Level** - Across experiments (in `loop.py`)
2. **CoSTEER Level** - Within single coding session (in `CoSTEER/__init__.py`)

---

## Changes Made (December 2025)

### 1. `rdagent/scenarios/data_science/loop.py`

#### Added `_get_score()` method with workspace fallback
```python
def _get_score(self, exp, feedback) -> Optional[float]:
    """
    Tries multiple sources:
    1. feedback.score attribute
    2. exp.result DataFrame
    3. scores.csv from workspace (fallback when result was cleared)
    """
```
**Why:** `HypothesisFeedback` doesn't have a `score` attribute, and `exp.result` can be `None` if cleared. This ensures we can detect if code produced a score.

#### Added comprehensive `[DEBUG_SKILL]` logging
- Logs when debug skill extraction check runs
- Logs pattern detection results
- Logs why patterns are/aren't detected

### 2. `rdagent/components/coder/CoSTEER/__init__.py`

#### Changed logging from `debug` to `info` level
All `[DEBUG_SKILL]` messages now visible in normal logs.

#### Added LLM-based transition detection
```python
def _is_debug_skill_worthy_transition(self, prev_fb, curr_fb, prev_code, curr_code):
    """Uses LLM to decide if transition is worthy of extraction"""
```
**Why:** More intelligent filtering - only extracts meaningful debug patterns, not trivial fixes.

---

## How Debug Skill Extraction Works

### Loop Level (Pattern 1: Consecutive)

```
Experiment N: Code crashes, no score produced → prev_failed=True
Experiment N+1: Code runs, produces score → current_success=True
→ Pattern detected! Extract debug skill.
```

**Key check:**
```python
current_success = self._has_valid_score(current_exp, current_feedback)
prev_failed = not self._has_valid_score(prev_exp, prev_feedback)

if current_success and prev_failed:
    # Extract debug skill
```

### CoSTEER Level (Within evolving_trace)

```
Step N: is_acceptable() = False (code failed)
Step N+1: is_acceptable() = True (code works)
→ LLM evaluates if this is a worthy transition
→ If YES, extract debug skill
```

---

## Log Messages to Look For

### Success Path (Debug Skill Extracted)

```
[DEBUG_SKILL] === Debug Skill Extraction Check ===
[DEBUG_SKILL] enable_debug_skill_extraction=True
[DEBUG_SKILL] has debug_skill_extractor=True
[DEBUG_SKILL] has global_kb=True
[DEBUG_SKILL] Calling _detect_failure_patterns...
[DEBUG_SKILL] Pattern detection called - current_score=0.75, history_len=3
[DEBUG_SKILL] Pattern 1: prev_score=None, prev_failed=True, current_score=0.75, current_success=True
🔍 Detected consecutive failure→success pattern
🐛 Detected consecutive pattern, extracting debug skill...
🔧 Learned debug skill: some_skill_name (severity: medium)
```

### Common Failure Cases

#### Case 1: No pattern detected (all experiments have scores)
```
[DEBUG_SKILL] Pattern 1: prev_score=0.72, prev_failed=False, current_score=0.75, current_success=True
[DEBUG_SKILL] No pattern detected - experiment_history has 5 items
```
**Meaning:** Previous experiment also produced a score (code ran), so no failure→success transition.

#### Case 2: Empty history
```
[DEBUG_SKILL] Pattern detection called - current_score=0.75, history_len=0
[DEBUG_SKILL] No pattern detected - experiment_history has 0 items
```
**Meaning:** First experiment, no previous to compare.

#### Case 3: Debug skill extraction disabled
```
[DEBUG_SKILL] Skipped: enable=False, extractor=True, kb=True
```
**Meaning:** `enable_debug_skill_extraction=False` in config.

#### Case 4: Missing components
```
[DEBUG_SKILL] Skipping debug skill extraction: global_kb or debug_skill_extractor is None
```
**Meaning:** Setup issue - components not passed to coder.

#### Case 5: CoSTEER - Not enough steps
```
[DEBUG_SKILL] Skipping debug skill extraction: only 1 steps (need >= 2)
```
**Meaning:** CoSTEER finished in 1 iteration, no transitions to analyze.

#### Case 6: LLM rejects transition
```
[DEBUG_SKILL] Step 3, task 0: LLM decision=False, reason=NO - This is a minor formatting change, not a meaningful bug fix
```
**Meaning:** LLM decided this transition isn't worthy of extraction.

#### Case 7: Extractor returns None
```
[DEBUG_SKILL] Pattern result: consecutive
🐛 Detected consecutive pattern, extracting debug skill...
[DEBUG_SKILL] Extractor returned None for consecutive pattern
```
**Meaning:** Pattern detected, but `debug_skill_extractor.extract_from_transition()` failed or returned None.

---

## Validation Checklist

Run your experiment and check:

### 1. Is extraction being attempted?
Look for: `[DEBUG_SKILL] === Debug Skill Extraction Check ===`
- If missing: Check if `record()` method is being called

### 2. Are all components present?
Look for:
```
[DEBUG_SKILL] enable_debug_skill_extraction=True
[DEBUG_SKILL] has debug_skill_extractor=True
[DEBUG_SKILL] has global_kb=True
```
- If any is `False`: Check initialization in `DataScienceRDLoop.__init__`

### 3. Are failure patterns occurring?
Look for: `[DEBUG_SKILL] Pattern 1: prev_score=..., prev_failed=..., current_score=..., current_success=...`
- If `prev_failed=False` always: All experiments run successfully (produce scores)
- This means Pattern 1 won't trigger (it needs code crash → code works)

### 4. Is CoSTEER extraction working?
Look for: `[DEBUG_SKILL] _extract_debug_skills_from_trace called`
- Then check: `[DEBUG_SKILL] Step N: LLM decision=True/False`
- If all `decision=False`: LLM is rejecting all transitions as not worthy

### 5. Are skills being saved?
Look for: `🔧 Learned debug skill: ... (severity: ...)`
- If missing after pattern detected: Check `global_kb.add_or_update_debug_skill()`

---

## Files Changed

| File | Changes |
|------|---------|
| `rdagent/scenarios/data_science/loop.py` | Added `_get_score()`, comprehensive logging |
| `rdagent/components/coder/CoSTEER/__init__.py` | LLM-based detection, `info` level logging |

---

## Configuration

In `rdagent/app/data_science/conf.py`:
```python
enable_debug_skill_extraction: bool = True  # Must be True
debug_skill_history_window: int = 10  # How many experiments to keep in history
```

---

## Troubleshooting

### "No debug skills extracted but experiments are running"

1. Check if experiments are crashing (producing no scores)
   - If all produce scores → Pattern 1 won't trigger
   - Pattern 1 needs: no score → has score

2. Check CoSTEER logs for LLM decisions
   - If all `decision=False` → LLM is too strict
   - Consider adjusting the prompt in `_is_debug_skill_worthy_transition()`

### "Pattern detected but no skill saved"

1. Check if extractor returned None
   - Look for: `[DEBUG_SKILL] Extractor returned None`
   - Check `debug_extractor.py` for validation logic

2. Check for exceptions
   - Look for: `Error extracting debug skill:`

---

## Expected Behavior

When working correctly, you should see:

1. **Every experiment:** `[DEBUG_SKILL] === Debug Skill Extraction Check ===`
2. **Pattern detection:** `[DEBUG_SKILL] Pattern 1: prev_score=..., prev_failed=..., ...`
3. **When failure→success occurs:** `🔍 Detected consecutive failure→success pattern`
4. **Skill learned:** `🔧 Learned debug skill: ... (severity: ...)`

The debug skills are saved to: `~/.rdagent/global_knowledge/debugging_skills/`

---

*Last updated: December 2025*
