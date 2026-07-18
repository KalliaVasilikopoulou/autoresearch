"""Agent 1: Training Specialist - Decides hyperparameters and trains models."""

import os
import random
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json


class Agent1TrainingSpecialist:
    """Trains models and adjusts hyperparameters based on agent feedback."""

    def __init__(self, config_path: str = "agents_config.yaml"):
        self.config = self._load_config(config_path)
        self.agent1_config = self.config.get("agent1", {})
        self.use_llm = self.agent1_config.get("use_llm", False)
        self.accuracy_threshold = self.agent1_config.get("accuracy_threshold", 0.95)
        self.cost_limit_usd = self.agent1_config.get("cost_limit_usd", 50.0)
        self.training_budget = self.agent1_config.get("training_budget_seconds", 300)

        self.model_config_path = Path("model_hyperparams.yaml")
        self.current_hyperparams = self._init_hyperparams()
        self.total_api_cost = 0.0
        self.best_val_bpb = float("inf")

        self.claude = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _init_hyperparams(self) -> Dict[str, Any]:
        """Initialize or load hyperparameters."""
        if self.model_config_path.exists():
            with open(self.model_config_path, "r") as f:
                return yaml.safe_load(f) or self._default_hyperparams()
        return self._default_hyperparams()

    def _default_hyperparams(self) -> Dict[str, Any]:
        """Default hyperparameters for GPT model."""
        return {
            "n_layer": 12,  # Number of layers
            "n_head": 8,  # Number of attention heads
            "n_embd": 512,  # Embedding dimension
            "learning_rate": 1e-3,
            "batch_size": 128,
            "warmup_ratio": 0.1,
            "weight_decay": 0.1,
        }

    def _save_hyperparams(self):
        """Save current hyperparams to YAML."""
        with open(self.model_config_path, "w") as f:
            yaml.dump(self.current_hyperparams, f)

    def decide_next_hyperparams(
        self,
        latest_summary: Optional[str] = None,
        evidence: Optional[list] = None,
        stuck_signal: bool = False,
        latest_val_bpb: Optional[float] = None,
        iteration: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Decide next hyperparameters using heuristics (+ optional Claude).

        Returns:
            New hyperparams dict, or None to STOP
        """

        # Check stopping conditions
        if latest_val_bpb is not None:
            if latest_val_bpb < self.accuracy_threshold:
                print(f"[Agent 1] Accuracy threshold reached: {latest_val_bpb:.6f}")
                return None

            if latest_val_bpb < self.best_val_bpb:
                self.best_val_bpb = latest_val_bpb

        if self.total_api_cost >= self.cost_limit_usd:
            print(f"[Agent 1] Cost limit exceeded: ${self.total_api_cost:.2f}")
            return None

        print(f"\n[Agent 1] Deciding next hyperparameters (iteration {iteration})...")

        # PRIMARY: use report-driven evidence when available
        new_hyperparams = self._evidence_adjustment(
            latest_summary=latest_summary,
            evidence=evidence,
            stuck_signal=stuck_signal,
            iteration=iteration,
        )

        # FALLBACK: use heuristics if evidence is sparse
        if not evidence:
            new_hyperparams = self._heuristic_adjustment(
                latest_summary, stuck_signal, iteration
            )

        # OPTIONAL: If LLM enabled, get Claude suggestions
        if self.use_llm and latest_summary:
            try:
                self._init_claude()
                claude_suggestion = self._get_claude_suggestion(
                    new_hyperparams, latest_summary
                )
                # Claude can override or confirm heuristic suggestion
                new_hyperparams = claude_suggestion or new_hyperparams
                self.total_api_cost += 0.03  # Estimate
            except Exception as e:
                print(f"[Agent 1] Claude suggestion failed: {e}")

        self.current_hyperparams = new_hyperparams
        self._save_hyperparams()

        print(f"[Agent 1] Next hyperparams: {new_hyperparams}")
        return new_hyperparams

    def _evidence_adjustment(
        self,
        latest_summary: Optional[str],
        evidence: Optional[list],
        stuck_signal: bool,
        iteration: int,
    ) -> Dict[str, Any]:
        """Translate structured evidence from Agents 2 and 3 into new hyperparameters."""
        new_params = self.current_hyperparams.copy()

        if stuck_signal:
            print("[Agent 1] Model stuck - trying radical changes")
            new_params["n_layer"] = random.randint(8, 20)
            new_params["n_embd"] = random.choice([256, 384, 512, 768, 1024])
            return new_params

        if not evidence:
            return new_params

        importance_by_param: Dict[str, float] = {}
        for item in evidence:
            if not isinstance(item, dict):
                continue
            for param, score in item.get("hyperparameter_importance", {}).items():
                importance_by_param[param] = importance_by_param.get(param, 0.0) + float(score)
            if item.get("stuck_signal"):
                print("[Agent 1] Evidence indicates a stuck pattern")

        if latest_summary:
            summary_lower = latest_summary.lower()
            if "learning rate" in summary_lower and "important" in summary_lower:
                new_params["learning_rate"] *= 1.5
            if ("depth" in summary_lower or "layer" in summary_lower) and (
                "important" in summary_lower or "matter" in summary_lower
            ):
                new_params["n_layer"] = max(4, min(new_params["n_layer"] + 1, 24))

        if "learning_rate" in importance_by_param:
            new_params["learning_rate"] *= 1.2
        if "n_layer" in importance_by_param:
            new_params["n_layer"] = max(4, min(new_params["n_layer"] + 1, 24))
        if "n_embd" in importance_by_param:
            new_params["n_embd"] = int(new_params["n_embd"] * 1.1)

        return new_params

    def _heuristic_adjustment(
        self, summary: Optional[str], stuck: bool, iteration: int
    ) -> Dict[str, Any]:
        """
        Heuristic-based hyperparameter adjustment.

        Rules:
        - If stuck: Try radical changes (different depth, architecture)
        - Else: Make incremental adjustments based on importance
        """

        new_params = self.current_hyperparams.copy()

        if stuck:
            print("[Agent 1] Model stuck - trying radical changes")
            # Large change to depth
            new_params["n_layer"] = random.randint(8, 20)
            new_params["n_embd"] = random.choice([256, 384, 512, 768, 1024])
            return new_params

        # Extract insights from summary
        if summary:
            summary_lower = summary.lower()

            # Rule 1: If learning rate is mentioned as important
            if "learning rate" in summary_lower and "important" in summary_lower:
                lr_factor = random.choice([1.5, 2.0, 0.7, 0.5])  # Try 2x or 0.5x
                new_params["learning_rate"] *= lr_factor
                print(f"[Agent 1] Adjusted LR: {new_params['learning_rate']:.2e}")

            # Rule 2: If depth/layers mentioned as important
            if ("depth" in summary_lower or "layer" in summary_lower) and (
                "important" in summary_lower or "matter" in summary_lower
            ):
                new_params["n_layer"] += random.choice([-1, 1, 2])
                new_params["n_layer"] = max(4, min(new_params["n_layer"], 24))
                print(f"[Agent 1] Adjusted layers: {new_params['n_layer']}")

            # Rule 3: If embedding size mentioned
            if "embedding" in summary_lower and "important" in summary_lower:
                emb_factor = random.choice([0.8, 1.2, 1.5])
                new_params["n_embd"] = int(new_params["n_embd"] * emb_factor)
                print(f"[Agent 1] Adjusted embedding: {new_params['n_embd']}")
        else:
            # No summary yet - early iterations, try random exploration
            if iteration < 5:
                print("[Agent 1] Early iteration - random exploration")
                new_params["n_layer"] = random.randint(6, 18)
                new_params["learning_rate"] = 10 ** random.uniform(-4, -2)

        return new_params

    def train_model(
        self, hyperparams: Dict[str, Any], dry_run: bool = False, iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Train model and return metrics.

        Runs: uv run train.py (with hyperparams from YAML)
        """
        # Save hyperparams to YAML for train.py to read
        self._save_hyperparams()

        print(f"[Agent 1] Starting training with: {hyperparams}")

        if dry_run:
            print("[Agent 1] Dry run enabled; skipping actual training")
            return {
                "val_bpb": 1.0 - 0.001 * (iteration + 1),
                "training_time": 0.0,
                "checkpoint_path": None,
                "status": "dry_run",
            }

        try:
            # Run training subprocess
            result = subprocess.run(
                ["uv", "run", "train.py"],
                capture_output=True,
                text=True,
                timeout=self.training_budget + 30,  # Allow 30s overhead
            )

            # Parse output for metrics
            metrics = self._parse_training_output(result.stdout)
            print(f"[Agent 1] Training complete. Metrics: {metrics}")

            return metrics

        except subprocess.TimeoutExpired:
            print("[Agent 1] Training timeout")
            return {"val_bpb": float("inf"), "error": "timeout"}
        except Exception as e:
            print(f"[Agent 1] Training error: {e}")
            return {"val_bpb": float("inf"), "error": str(e)}

    def _parse_training_output(self, stdout: str) -> Dict[str, Any]:
        """Parse metrics from train.py output."""
        metrics = {
            "val_bpb": float("inf"),
            "train_loss": None,
            "train_time": None,
        }

        # Look for specific patterns in output
        lines = stdout.split("\n")
        for line in lines:
            if "val_bpb" in line.lower():
                try:
                    # Extract number
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "bpb" in part.lower() and i + 1 < len(parts):
                            metrics["val_bpb"] = float(parts[i + 1])
                except:
                    pass

        return metrics

    def _init_claude(self):
        """Lazy-load Claude client."""
        if self.claude is not None:
            return

        try:
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.claude = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed")

    def _get_claude_suggestion(
        self, heuristic_params: Dict[str, Any], summary: str
    ) -> Optional[Dict[str, Any]]:
        """Get Claude to suggest hyperparameters."""
        prompt = f"""Based on this summary, review our heuristic hyperparameter suggestion:

Summary insights:
{summary[:500]}

Our heuristic suggestion:
{heuristic_params}

Should we adjust this? Provide JSON with adjustments (or empty {{}} if heuristic is good).
Example: {{"n_layer": 14, "learning_rate": 0.002}}"""

        try:
            message = self.claude.messages.create(
                model="claude-opus-4-7",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            # Try to parse JSON from response
            response_text = message.content[0].text
            # Simple JSON extraction
            if "{" in response_text and "}" in response_text:
                json_str = (
                    response_text[response_text.find("{") : response_text.rfind("}") + 1]
                )
                adjustments = json.loads(json_str)
                if adjustments:
                    updated = heuristic_params.copy()
                    updated.update(adjustments)
                    print(f"[Agent 1] Claude suggested adjustments: {adjustments}")
                    return updated

            return None
        except Exception as e:
            print(f"[Agent 1] Claude suggestion error: {e}")
            return None
