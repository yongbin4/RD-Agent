"""SOTA (State-of-the-Art) model storage and management."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil


@dataclass
class SOTAModel:
    """A state-of-the-art model from a competition."""

    competition: str
    rank: int  # 1, 2, 3
    score: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Code
    code_files: Dict[str, str] = field(default_factory=dict)  # filename → code content
    workspace_path: Optional[Path] = None  # Full workspace backup path

    # Metadata
    hypothesis: str = ""
    skills_used: List[str] = field(default_factory=list)
    approach_description: str = ""

    def save_to_directory(self, base_path: Path):
        """Save SOTA model to disk."""
        # Create directory: rank_{rank}_score_{score}/
        dir_name = f"rank_{self.rank}_score_{self.score:.6f}"
        sota_dir = base_path / dir_name
        sota_dir.mkdir(parents=True, exist_ok=True)

        # Save code files
        for filename, code in self.code_files.items():
            (sota_dir / filename).write_text(code)

        # Save metadata
        metadata = {
            "competition": self.competition,
            "rank": self.rank,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "hypothesis": self.hypothesis,
            "skills_used": self.skills_used,
            "approach_description": self.approach_description,
            "code_files": list(self.code_files.keys()),
        }
        (sota_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Backup full workspace if provided
        if self.workspace_path and self.workspace_path.exists():
            workspace_backup = sota_dir / "workspace"
            if workspace_backup.exists():
                shutil.rmtree(workspace_backup)
            shutil.copytree(self.workspace_path, workspace_backup)

    @classmethod
    def load_from_directory(cls, sota_dir: Path) -> "SOTAModel":
        """Load SOTA model from disk."""
        metadata_path = sota_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {sota_dir}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load code files
        code_files = {}
        for filename in metadata.get("code_files", []):
            code_path = sota_dir / filename
            if code_path.exists():
                code_files[filename] = code_path.read_text()

        # Get workspace path if exists
        workspace_path = sota_dir / "workspace"
        if not workspace_path.exists():
            workspace_path = None

        return cls(
            competition=metadata["competition"],
            rank=metadata["rank"],
            score=metadata["score"],
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            code_files=code_files,
            workspace_path=workspace_path,
            hypothesis=metadata.get("hypothesis", ""),
            skills_used=metadata.get("skills_used", []),
            approach_description=metadata.get("approach_description", ""),
        )

    @classmethod
    def from_experiment(cls, experiment, feedback, rank: int = 1) -> "SOTAModel":
        """Convert an experiment to SOTA model."""
        from rdagent.core.experiment import Experiment

        # Extract code files from workspace
        code_files = {}
        if hasattr(experiment, "experiment_workspace") and experiment.experiment_workspace:
            workspace = experiment.experiment_workspace
            if hasattr(workspace, "file_dict"):
                code_files = workspace.file_dict.copy()

        # Extract workspace path
        workspace_path = None
        if hasattr(experiment, "experiment_workspace") and experiment.experiment_workspace:
            if hasattr(experiment.experiment_workspace, "workspace_path"):
                workspace_path = Path(experiment.experiment_workspace.workspace_path)

        # Extract hypothesis info
        hypothesis = ""
        if hasattr(experiment, "hypothesis"):
            hypothesis = str(experiment.hypothesis)

        # Extract score from feedback
        score = 0.0
        if hasattr(feedback, "score"):
            score = float(feedback.score)
        elif hasattr(feedback, "final_decision"):
            # Try to extract score from observations
            score = 0.0  # Default

        return cls(
            competition="unknown",  # Will be set by caller
            rank=rank,
            score=score,
            code_files=code_files,
            workspace_path=workspace_path,
            hypothesis=hypothesis,
            approach_description=hypothesis,
        )

    def to_experiment(self):
        """Convert SOTA back to Experiment object for resuming.

        This creates a minimal experiment that can be used as a starting point.
        """
        from rdagent.core.experiment import Experiment, Hypothesis
        from rdagent.core.workspace import FBWorkspace

        # Reconstruct hypothesis
        hypothesis = Hypothesis(
            hypothesis=self.approach_description,
            reason=f"SOTA solution with score {self.score}",
            concise_reason=f"SOTA baseline (rank {self.rank})",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
        )

        # Reconstruct workspace
        workspace = FBWorkspace()
        if self.code_files:
            workspace.inject_files(**self.code_files)

        # Create experiment
        exp = Experiment(
            hypothesis=hypothesis,
            sub_tasks=[],
            sub_workspace_list=[],
            experiment_workspace=workspace,
        )

        return exp
