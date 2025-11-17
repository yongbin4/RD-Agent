"""Skill data structures for storing and managing reusable patterns."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import hashlib


@dataclass
class SkillExample:
    """A concrete example of skill usage from a specific competition."""

    competition: str
    code: str
    context: str
    score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "competition": self.competition,
            "code": self.code,
            "context": self.context,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillExample":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Skill:
    """A reusable pattern extracted from successful experiments."""

    name: str  # Short identifier (e.g., "missing_value_imputation")
    description: str  # Human-readable description of what it does
    code_pattern: str  # The key code snippet
    applicable_contexts: List[str]  # Tags like ["tabular", "missing_values"]
    examples: List[SkillExample] = field(default_factory=list)

    # Statistics
    success_count: int = 0
    attempt_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    version: int = 1

    # Metadata
    source_competitions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    id: str = field(default="")

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            # Generate ID from name and code pattern hash
            content = f"{self.name}:{self.code_pattern}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:8]

    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success_count / max(1, self.attempt_count)

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        md = f"""# Skill: {self.name.replace('_', ' ').title()}

**ID**: {self.id}
**Created**: {self.created_at.strftime('%Y-%m-%d')}
**Success Rate**: {self.success_rate():.1%} ({self.success_count}/{self.attempt_count} experiments)
**Applicable Contexts**: {', '.join(self.applicable_contexts)}

## Description
{self.description}

## When to Use
This skill is applicable in contexts involving: {', '.join(self.applicable_contexts)}

## Code Pattern
```python
{self.code_pattern}
```

## Examples

"""
        for i, example in enumerate(self.examples, 1):
            md += f"""### Example {i}: {example.competition}
- **Context**: {example.context}
- **Result**: Score {example.score:.4f}
- **Date**: {example.timestamp.strftime('%Y-%m-%d')}

```python
{example.code}
```

"""

        md += f"""## Version History
- v{self.version}.0 ({self.created_at.strftime('%Y-%m-%d')}): {len(self.examples)} examples from {len(set(ex.competition for ex in self.examples))} competitions
"""

        if self.last_used:
            md += f"\n**Last Used**: {self.last_used.strftime('%Y-%m-%d')}\n"

        return md

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "code_pattern": self.code_pattern,
            "applicable_contexts": self.applicable_contexts,
            "examples": [ex.to_dict() for ex in self.examples],
            "success_count": self.success_count,
            "attempt_count": self.attempt_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "version": self.version,
            "source_competitions": self.source_competitions,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_used"):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        data["examples"] = [SkillExample.from_dict(ex) for ex in data.get("examples", [])]
        return cls(**data)

    @classmethod
    def from_directory(cls, skill_dir: Path) -> "Skill":
        """Load skill from directory containing README.md, examples, and metadata.json."""
        metadata_path = skill_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {skill_dir}")

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_stats(self, success: bool):
        """Update skill statistics after usage."""
        self.attempt_count += 1
        if success:
            self.success_count += 1
        self.last_used = datetime.now()
