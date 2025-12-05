# TESTING SKILL EXTRACTION SYSTEM

## ⚠️ TEMPORARY TESTING MODE ACTIVE

This document tracks temporary changes made to test the skill extraction system.

## Changes Made

### 1. Modified Skill Extraction Policy (`rdagent/scenarios/data_science/loop.py`)

**Lines 313-342**: `_should_extract_skill_from_score()` method
- **Original**: Only extracts skills when score improves over previous best
- **Testing**: Always returns `True` - extracts skills from EVERY experiment
- **Purpose**: Verify skill extraction works regardless of experiment success

**Lines 459-491**: Enhanced logging in `record()` method
- Added detailed logging for skill extraction process
- Shows when extraction triggers, what skills are extracted, and save status
- Helps verify the system is working

### 2. Added GPU Disable Settings (`.env`)

**Lines 23-24**:
```bash
DS_DOCKER_ENABLE_GPU=false
MLEB_DOCKER_ENABLE_GPU=false
```
- Allows Docker containers to run without nvidia-docker
- Fixes container creation failures

### 3. Added Submission Format Info (`rdagent/scenarios/data_science/scen/__init__.py`)

**Lines 260-273**: `get_sample_submission_format()` method
- Extracts column names from sample_submission.csv
- Injects into prompts to prevent format errors

**Updated**: `prompts.yaml` and template rendering calls

## How to Verify It's Working

### During a Run, Look For:

1. **Skill Extraction Logs:**
   ```
   🔍 SKILL EXTRACTION CHECK (Loop X)
   🧪 TESTING MODE: Extracting skill regardless of score
   📊 Skill extraction triggered: TESTING MODE
   ```

2. **Saved Skills:**
   ```
   [1/N] Saving skill: Feature Engineering with StandardScaler
     - Success rate: 100.0%
     - Contexts: ['tabular', 'classification']
     ✅ Saved to global KB
   ```

3. **Files Created:**
   - Check `~/.rdagent/global_knowledge/skills/` for new directories
   - Each skill gets a `skill_{id}_{name}/` directory

4. **Skill Retrieval (in subsequent loops):**
   - Look for logs about querying skills
   - Check if skills appear in code generation prompts

## How to Restore Original Behavior

### After Testing, Revert Changes:

1. **Restore skill extraction policy:**
   Edit `/home/ubuntu/RD-Agent/rdagent/scenarios/data_science/loop.py`
   - **Remove lines 316-323** (testing mode)
   - **Uncomment lines 325-342** (original logic)

2. **Remove extra logging (optional):**
   - Lines 462-467 (extraction check header)
   - Lines 472, 478-485 (detailed save logging)
   - Keep if you want verbose logging

3. **GPU settings (keep or remove):**
   - If you install nvidia-docker, you can remove the GPU disable settings
   - Or keep them if running CPU-only

## Quick Revert Command

```bash
# Go to the modified lines
vim +313 /home/ubuntu/RD-Agent/rdagent/scenarios/data_science/loop.py

# Delete lines 316-323 (testing mode)
# Uncomment lines 325-342 (original logic)
```

## Testing Checklist

- [ ] Run experiment and see skill extraction logs
- [ ] Verify skills are saved to `~/.rdagent/global_knowledge/skills/`
- [ ] Check skill README.md files are created
- [ ] Verify skills can be loaded (check next loop logs)
- [ ] Confirm skills appear in prompts when relevant
- [ ] Test skill querying works (relevant skills retrieved)

## Notes

- Even with permissive extraction, code still needs to execute (not just create experiments)
- Docker containers must start successfully (GPU fix required)
- Skills might be low quality from failed experiments, but that's OK for testing
- Once verified system works, restore original policy and focus on improving experiment success rate
