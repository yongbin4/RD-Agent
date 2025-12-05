"""Debug skill data structures for storing problem-solving patterns."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import hashlib


@dataclass
class DebugExample:
    """A concrete example of a problem-solution pattern from a specific competition."""

    competition: str
    symptom: str  # How the problem manifested (error message, poor metrics, etc.)
    failed_code: str  # The code that didn't work
    solution_code: str  # The fixed code
    context: str  # Task context when this occurred
    before_score: Optional[float]  # Score before fix (if applicable)
    after_score: Optional[float] = None  # Score after fix (None if unknown)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "competition": self.competition,
            "symptom": self.symptom,
            "failed_code": self.failed_code,
            "solution_code": self.solution_code,
            "context": self.context,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebugExample":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class DebugSkill:
    """A problem-solving pattern extracted from failure-to-success transitions."""

    name: str  # Short identifier (e.g., "data_leakage_time_series")
    description: str  # Human-readable description of the problem and solution

    # Problem definition
    symptom: str  # How to recognize this problem (error messages, metric patterns, etc.)
    root_cause: str  # Why this problem occurs (conceptual explanation)

    # Solution
    failed_approach: str  # Common code pattern that causes this problem
    solution: str  # Code pattern that fixes the problem

    # Metadata
    applicable_contexts: List[str]  # Tags like ["time_series", "data_leakage", "preprocessing"]
    examples: List[DebugExample] = field(default_factory=list)

    # Statistics
    detection_count: int = 0  # How many times this problem was encountered
    fix_success_count: int = 0  # How many times the solution worked
    created_at: datetime = field(default_factory=datetime.now)
    last_encountered: Optional[datetime] = None
    version: int = 1

    # Additional metadata
    source_competitions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high (how bad is this problem)
    id: str = field(default="")

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            # Generate ID from name and symptom hash (16 chars to reduce collisions)
            content = f"{self.name}:{self.symptom}:{self.root_cause}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:16]

    def fix_success_rate(self) -> float:
        """Calculate how often this solution successfully fixes the problem."""
        return self.fix_success_count / max(1, self.detection_count)

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        md = f"""# Debug Skill: {self.name.replace('_', ' ').title()}

**ID**: {self.id}
**Severity**: {self.severity.upper()}
**Created**: {self.created_at.strftime('%Y-%m-%d')}
**Fix Success Rate**: {self.fix_success_rate():.1%} ({self.fix_success_count}/{self.detection_count} times)
**Applicable Contexts**: {', '.join(self.applicable_contexts)}

## Description
{self.description}

## Problem Recognition

### Symptom
{self.symptom}

### Root Cause
{self.root_cause}

## Solution

### ❌ Failed Approach (What NOT to do)
```python
{self.failed_approach}
```

### ✅ Correct Approach (How to fix it)
```python
{self.solution}
```

## Examples

"""
        for i, example in enumerate(self.examples, 1):
            score_change = ""
            if example.before_score is not None and example.after_score is not None:
                score_change = f" (improved from {example.before_score:.4f} to {example.after_score:.4f})"

            after_score_str = f"{example.after_score:.4f}" if example.after_score is not None else "N/A"
            md += f"""### Example {i}: {example.competition}
- **Context**: {example.context}
- **Symptom**: {example.symptom}
- **Result**: Score {after_score_str}{score_change}
- **Date**: {example.timestamp.strftime('%Y-%m-%d')}

**Failed Code**:
```python
{example.failed_code}
```

**Fixed Code**:
```python
{example.solution_code}
```

"""

        md += f"""## Version History
- v{self.version}.0 ({self.created_at.strftime('%Y-%m-%d')}): {len(self.examples)} examples from {len(set(ex.competition for ex in self.examples))} competitions
"""

        if self.last_encountered:
            md += f"\n**Last Encountered**: {self.last_encountered.strftime('%Y-%m-%d')}\n"

        return md

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "symptom": self.symptom,
            "root_cause": self.root_cause,
            "failed_approach": self.failed_approach,
            "solution": self.solution,
            "applicable_contexts": self.applicable_contexts,
            "examples": [ex.to_dict() for ex in self.examples],
            "detection_count": self.detection_count,
            "fix_success_count": self.fix_success_count,
            "created_at": self.created_at.isoformat(),
            "last_encountered": self.last_encountered.isoformat() if self.last_encountered else None,
            "version": self.version,
            "source_competitions": self.source_competitions,
            "tags": self.tags,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebugSkill":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_encountered"):
            data["last_encountered"] = datetime.fromisoformat(data["last_encountered"])
        data["examples"] = [DebugExample.from_dict(ex) for ex in data.get("examples", [])]
        return cls(**data)

    @classmethod
    def from_directory(cls, skill_dir: Path) -> "DebugSkill":
        """Load debug skill from directory containing README.md and metadata.json."""
        metadata_path = skill_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {skill_dir}")

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_stats(self, fixed: bool):
        """Update debug skill statistics after encountering this problem."""
        self.detection_count += 1
        if fixed:
            self.fix_success_count += 1
        self.last_encountered = datetime.now()
