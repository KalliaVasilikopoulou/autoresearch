"""State management for multi-agent system."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch


class StateManager:
    """Tracks models, reports, hyperparameters, and iteration state."""

    def __init__(self, state_dir: str = "./state"):
        self.state_dir = Path(state_dir)
        self.models_dir = self.state_dir / "models"
        self.metadata_file = self.state_dir / "metadata.json"
        self.report_tracking_file = self.state_dir / "report_tracking.json"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_or_init("metadata")
        self.report_tracking = self._load_or_init("report_tracking")

    def _load_or_init(self, file_type: str) -> Dict[str, Any]:
        """Load existing state or initialize empty dict."""
        if file_type == "metadata":
            file = self.metadata_file
            default = {"models": {}, "latest_summary": None, "iteration": 0}
        else:  # report_tracking
            file = self.report_tracking_file
            default = {"reports": {}, "batch_count": 0, "latest_summary_covers_up_to": 0}

        if file.exists():
            try:
                with open(file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return default
        return default

    def _save_metadata(self):
        """Persist metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _save_report_tracking(self):
        """Persist report tracking to disk."""
        with open(self.report_tracking_file, "w") as f:
            json.dump(self.report_tracking, f, indent=2)

    def save_model(self, model, iteration: int, hyperparams: Dict[str, Any]) -> str:
        """Save model checkpoint and track metadata."""
        model_id = f"model_{iteration:04d}"
        model_path = self.models_dir / f"{model_id}.pt"

        if model is not None:
            # Save weights only
            torch.save(model.state_dict(), model_path)

        # Track metadata
        self.metadata.setdefault("models", {})[model_id] = {
            "iteration": iteration,
            "hyperparams": hyperparams,
            "val_bpb": None,
            "report_id": None,
            "timestamp": time.time(),
        }
        self.metadata["iteration"] = iteration
        self._save_metadata()

        return model_id

    def load_model(self, model_id: str, model_class):
        """Load model checkpoint."""
        model_path = self.models_dir / f"{model_id}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model

    def link_model_to_report(self, model_id: str, report_id: str):
        """Associate trained model with its XAI report."""
        if model_id in self.metadata.setdefault("models", {}):
            self.metadata["models"][model_id]["report_id"] = report_id
            self._save_metadata()

    def update_val_bpb(self, model_id: str, val_bpb: float):
        """Update validation metric for a model."""
        if model_id in self.metadata.setdefault("models", {}):
            self.metadata["models"][model_id]["val_bpb"] = val_bpb
            self._save_metadata()

    def add_report(self, report_id: str, report_num: int, model_id: str):
        """Track new Agent 2 report."""
        self.report_tracking["reports"][report_id] = {
            "report_num": report_num,
            "model_id": model_id,
            "timestamp": time.time(),
        }
        self.report_tracking["batch_count"] += 1
        self._save_report_tracking()

    def should_create_summary(self, batch_size: int = 3) -> bool:
        """Check if enough reports for Agent 3 to create summary."""
        return self.report_tracking["batch_count"] >= batch_size

    def reset_batch_counter(self):
        """Reset after summary is created."""
        self.report_tracking["batch_count"] = 0
        self._save_report_tracking()

    def set_latest_summary(self, summary_id: str, covers_up_to_report: int):
        """Record new summary report."""
        self.metadata["latest_summary"] = summary_id
        self.report_tracking["latest_summary_covers_up_to"] = covers_up_to_report
        self._save_metadata()
        self._save_report_tracking()

    def get_latest_summary(self) -> Optional[str]:
        """Get most recent summary ID."""
        return self.metadata.get("latest_summary")

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked models."""
        return self.metadata["models"]

    def get_all_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked reports."""
        return self.report_tracking["reports"]

    def get_current_iteration(self) -> int:
        """Get current iteration counter."""
        return self.metadata.get("iteration", 0)

    def add_result(self, result: Dict[str, Any]):
        """Store a completed training result."""
        self.metadata.setdefault("results", []).append(result)
        self._save_metadata()

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all recorded training results."""
        return self.metadata.get("results", [])

    def add_evidence(self, evidence: Dict[str, Any]):
        """Store structured evidence from Agent 2."""
        self.metadata.setdefault("evidence", []).append(evidence)
        self._save_metadata()

    def get_recent_evidence(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent evidence items."""
        evidence = self.metadata.get("evidence", [])
        return evidence[-limit:]

    def add_summary(self, summary: Dict[str, Any]):
        """Store structured summary from Agent 3."""
        self.metadata.setdefault("summaries", []).append(summary)
        self._save_metadata()

    def get_latest_summary(self) -> Optional[str]:
        """Return the latest summary id that was recorded."""
        return self.metadata.get("latest_summary")
