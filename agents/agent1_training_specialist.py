"""Agent 1: Training Specialist - Decides hyperparameters and trains models."""

import math
import os
import random
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None


# The 4 learning-rate groups train.py's optimizer actually exposes (matches
# GPT.setup_optimizer's kwargs 1:1 — see train.py). Each has its own default
# and safe range since the groups operate on very different scales.
LR_KEYS = ("embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr")
LR_DEFAULTS = {
    "embedding_lr": 0.6,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.04,
    "scalar_lr": 0.5,
}
LR_SAFE_RANGES = {
    "embedding_lr": (0.05, 3.0),
    "unembedding_lr": (0.0005, 0.02),
    "matrix_lr": (0.005, 0.2),
    "scalar_lr": (0.05, 2.0),
}


class Agent1TrainingSpecialist:
    """Trains models and adjusts hyperparameters based on agent feedback."""

    def __init__(self, config_path: str = "agents_config.yaml"):
        self.config = self._load_config(config_path)
        self.agent1_config = self.config.get("agent1", {})
        self.use_llm = self.agent1_config.get("use_llm", False)
        self.accuracy_threshold = self.agent1_config.get("accuracy_threshold", 0.95)
        self.cost_limit_usd = self.agent1_config.get("cost_limit_usd", 50.0)
        self.training_budget = self.agent1_config.get("training_budget_seconds", 300)
        self.min_improvement = self.agent1_config.get("min_improvement", 0.005)
        self.max_stalled_iterations = self.agent1_config.get("max_stalled_iterations", 3)
        self.summary_strength = float(self.agent1_config.get("summary_strength", 2.0))
        self.lr_bounds: Dict[str, tuple] = {
            key: (
                float(self.agent1_config.get(f"{key}_min", LR_SAFE_RANGES[key][0])),
                float(self.agent1_config.get(f"{key}_max", LR_SAFE_RANGES[key][1])),
            )
            for key in LR_KEYS
        }

        self.model_config_path = Path("model_hyperparams.yaml")
        self.current_hyperparams = self._init_hyperparams()
        self.total_api_cost = 0.0
        self.best_val_bpb = float("inf")

        self.claude = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        if yaml is None:
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _init_hyperparams(self) -> Dict[str, Any]:
        """Initialize or load hyperparameters."""
        if self.model_config_path.exists():
            if yaml is not None:
                with open(self.model_config_path, "r") as f:
                    return yaml.safe_load(f) or self._default_hyperparams()
        return self._default_hyperparams()

    def _default_hyperparams(self) -> Dict[str, Any]:
        """Default hyperparameters for GPT model (mirrors train.py's baseline)."""
        return {
            "n_layer": 8,  # Number of layers
            "n_head": 4,  # Number of attention heads
            "n_embd": 512,  # Embedding dimension
            "embedding_lr": LR_DEFAULTS["embedding_lr"],
            "unembedding_lr": LR_DEFAULTS["unembedding_lr"],
            "matrix_lr": LR_DEFAULTS["matrix_lr"],
            "scalar_lr": LR_DEFAULTS["scalar_lr"],
            "batch_size": 8192,  # tokens per optimizer step (TOTAL_BATCH_SIZE)
            "warmup_ratio": 0.0,
            "weight_decay": 0.2,
        }

    def _save_hyperparams(self):
        """Save current hyperparams to YAML."""
        if yaml is None:
            return
        with open(self.model_config_path, "w") as f:
            yaml.dump(self.current_hyperparams, f)

    def decide_next_hyperparams(
        self,
        latest_summary: Optional[str] = None,
        evidence: Optional[list] = None,
        stuck_signal: bool = False,
        latest_val_bpb: Optional[float] = None,
        iteration: int = 0,
        recent_results: Optional[list] = None,
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

        if self._should_stop_early(recent_results=recent_results, latest_val_bpb=latest_val_bpb):
            print("[Agent 1] Stopping: no meaningful improvement over recent iterations")
            return None

        if self.total_api_cost >= self.cost_limit_usd:
            print(f"[Agent 1] Cost limit exceeded: ${self.total_api_cost:.2f}")
            return None

        print(f"\n[Agent 1] Deciding next hyperparameters (iteration {iteration})...")

        detected_stagnation = self._detect_stagnation(recent_results, latest_val_bpb)
        effective_stuck_signal = stuck_signal or detected_stagnation

        # PRIMARY: use report-driven evidence when available
        new_hyperparams = self._evidence_adjustment(
            latest_summary=latest_summary,
            evidence=evidence,
            stuck_signal=effective_stuck_signal,
            iteration=iteration,
        )

        # FALLBACK: use heuristics if evidence is sparse
        if not evidence:
            new_hyperparams = self._heuristic_adjustment(
                latest_summary, effective_stuck_signal, iteration
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

        # Claude's suggestion is free-form JSON — re-clamp every LR group
        # before it's ever saved, since that path can otherwise bypass the
        # safety bounds the rest of this class enforces everywhere else.
        for key in LR_KEYS:
            if key in new_hyperparams:
                new_hyperparams[key] = self._clamp_lr(key, float(new_hyperparams[key]))

        self.current_hyperparams = new_hyperparams
        self._save_hyperparams()

        print(f"[Agent 1] Next hyperparams: {new_hyperparams}")
        return new_hyperparams

    def _should_stop_early(
        self,
        recent_results: Optional[list],
        latest_val_bpb: Optional[float],
    ) -> bool:
        """Stop once the recent trend has stalled for enough iterations."""
        if not recent_results:
            return False

        values = []
        for item in recent_results:
            if not isinstance(item, dict):
                continue
            val_bpb = item.get("val_bpb")
            if val_bpb is None:
                continue
            try:
                values.append(float(val_bpb))
            except (TypeError, ValueError):
                continue

        if len(values) < self.max_stalled_iterations + 1:
            return False

        finite_values = [value for value in values if math.isfinite(value)]
        if len(finite_values) < self.max_stalled_iterations + 1:
            return False

        recent_window = finite_values[-(self.max_stalled_iterations + 1):]
        best_value = min(recent_window)
        latest_value = recent_window[-1]
        if latest_val_bpb is not None and math.isfinite(latest_val_bpb):
            latest_value = latest_val_bpb

        improvement = best_value - latest_value
        return improvement < self.min_improvement

    def _detect_stagnation(
        self,
        recent_results: Optional[list],
        latest_val_bpb: Optional[float],
    ) -> bool:
        """Return True when the recent validation trend shows little or no improvement."""
        if not recent_results:
            return False

        values = []
        for item in recent_results:
            if not isinstance(item, dict):
                continue
            val_bpb = item.get("val_bpb")
            if val_bpb is None:
                continue
            try:
                values.append(float(val_bpb))
            except (TypeError, ValueError):
                continue

        if len(values) < 3:
            return False

        finite_values = [value for value in values if math.isfinite(value)]
        if len(finite_values) < 3:
            return False

        window = finite_values[-3:]
        if window[0] == float("inf") or window[1] == float("inf") or window[2] == float("inf"):
            return False

        improved = window[-1] < window[-2] - 0.01 or window[-2] < window[-3] - 0.01
        if latest_val_bpb is not None and math.isfinite(latest_val_bpb):
            return not improved and latest_val_bpb >= min(window[-2], window[-3]) - 0.01
        return not improved

    def _radical_change(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Large random architecture jump used when the model looks stuck."""
        print("[Agent 1] Model stuck - trying radical changes")
        new_params["n_layer"] = random.randint(8, 20)
        new_params["n_embd"] = random.choice([256, 384, 512, 768, 1024])
        for key in LR_KEYS:
            new_params[key] = self._clamp_lr(key, float(new_params.get(key, LR_DEFAULTS[key])))
        return new_params

    def _nudge_lr(self, new_params: Dict[str, Any], key: str, direction: int, magnitude: float, scale: float = 0.20) -> None:
        """Multiplicative nudge for a single LR group, in-place."""
        lr = float(new_params.get(key, LR_DEFAULTS[key]))
        lr *= math.exp(direction * scale * magnitude)
        new_params[key] = self._clamp_lr(key, lr)

    def _pull_lr_toward(self, new_params: Dict[str, Any], key: str, target: float, pull: float) -> None:
        """Blend a single LR group toward a target value, in-place."""
        current = float(new_params.get(key, LR_DEFAULTS[key]))
        target = self._clamp_lr(key, target)
        new_params[key] = self._clamp_lr(key, current * (1.0 - pull) + target * pull)

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
            return self._radical_change(new_params)

        if not evidence:
            return new_params

        importance_by_param: Dict[str, float] = {}
        evidence_count_by_param: Dict[str, int] = {}
        for item in evidence:
            if not isinstance(item, dict):
                continue
            for param, score in item.get("hyperparameter_importance", {}).items():
                importance_by_param[param] = importance_by_param.get(param, 0.0) + float(score)
                evidence_count_by_param[param] = evidence_count_by_param.get(param, 0) + 1
            if item.get("stuck_signal"):
                print("[Agent 1] Evidence indicates a stuck pattern")

        avg_importance: Dict[str, float] = {}
        for param, total_score in importance_by_param.items():
            count = max(1, evidence_count_by_param.get(param, 1))
            avg_importance[param] = max(0.0, min(1.0, total_score / count))

        # Report-level weighted nudges (single-run evidence): smaller base step.
        for param, score in avg_importance.items():
            magnitude = self._importance_magnitude(score)
            direction = 1 if score >= 0.5 else -1

            if param in LR_KEYS:
                self._nudge_lr(new_params, param, direction, magnitude, scale=0.20)

            elif param == "n_layer":
                current = int(new_params.get("n_layer", 12))
                delta = int(round(direction * (1.0 + magnitude)))
                new_params["n_layer"] = max(4, min(current + delta, 24))

            elif param == "n_embd":
                current = int(new_params.get("n_embd", 512))
                factor = 1.0 + direction * (0.08 * magnitude)
                new_params["n_embd"] = min(int(current * factor), 1024)

            elif param == "n_head":
                current = int(new_params.get("n_head", 8))
                delta = int(round(direction * magnitude))
                new_params["n_head"] = max(1, min(current + delta, 16))

        # Summary-level updates: stronger than report-level (2x by default).
        if latest_summary:
            recs = self._extract_summary_recommendations(latest_summary)
            summary_hints = self._summary_importance_hints(latest_summary)
            summary_multiplier = max(1.0, self.summary_strength)

            for param in [*LR_KEYS, "n_layer", "n_embd", "n_head"]:
                if not summary_hints.get(param) and param not in recs:
                    continue

                score = avg_importance.get(param, 0.75)
                magnitude = self._importance_magnitude(score)
                direction = 1 if score >= 0.5 else -1

                if param in LR_KEYS:
                    if param in recs:
                        pull = min(0.8, 0.25 * summary_multiplier)
                        self._pull_lr_toward(new_params, param, float(recs[param]), pull)
                    else:
                        self._nudge_lr(new_params, param, direction, magnitude, scale=0.20 * summary_multiplier)

                elif param == "n_layer":
                    current = int(new_params.get("n_layer", 12))
                    if param in recs:
                        target = int(round(float(recs[param])))
                        step = int(round((target - current) * min(1.0, 0.35 * summary_multiplier)))
                        if step == 0 and target != current:
                            step = 1 if target > current else -1
                        current += step
                    else:
                        current += int(round(direction * summary_multiplier * (1.0 + magnitude)))
                    new_params["n_layer"] = max(4, min(current, 24))

                elif param == "n_embd":
                    current = int(new_params.get("n_embd", 512))
                    if param in recs:
                        target = max(128, int(round(float(recs[param]))))
                        pull = min(0.8, 0.25 * summary_multiplier)
                        current = int(current * (1.0 - pull) + target * pull)
                    else:
                        factor = 1.0 + direction * (0.08 * summary_multiplier * magnitude)
                        current = int(current * factor)
                    new_params["n_embd"] = min(max(128, current), 1024)

                elif param == "n_head":
                    current = int(new_params.get("n_head", 8))
                    if param in recs:
                        target = int(round(float(recs[param])))
                        step = int(round((target - current) * min(1.0, 0.35 * summary_multiplier)))
                        if step == 0 and target != current:
                            step = 1 if target > current else -1
                        current += step
                    else:
                        current += int(round(direction * summary_multiplier * magnitude))
                    new_params["n_head"] = max(1, min(current, 16))

        # Hard safety bound for LR groups after all evidence/summary adjustments.
        for key in LR_KEYS:
            new_params[key] = self._clamp_lr(key, float(new_params.get(key, LR_DEFAULTS[key])))

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
            return self._radical_change(new_params)

        # Extract insights from summary
        if summary:
            recs = self._extract_summary_recommendations(summary)
            hints = self._summary_importance_hints(summary)
            strength = max(1.0, self.summary_strength)

            for lr_key in LR_KEYS:
                if lr_key in recs:
                    pull = min(0.8, 0.25 * strength)
                    self._pull_lr_toward(new_params, lr_key, float(recs[lr_key]), pull)
                    print(f"[Agent 1] Summary-guided {lr_key}: {new_params[lr_key]:.2e}")
                elif hints.get(lr_key):
                    self._nudge_lr(new_params, lr_key, direction=1, magnitude=1.0, scale=0.25 * strength)
                    print(f"[Agent 1] Adjusted {lr_key}: {new_params[lr_key]:.2e}")

            if "n_layer" in recs:
                current_layer = int(new_params.get("n_layer", 12))
                target_layer = int(round(float(recs["n_layer"])))
                delta = int(round((target_layer - current_layer) * min(1.0, 0.35 * strength)))
                if delta == 0 and target_layer != current_layer:
                    delta = 1 if target_layer > current_layer else -1
                new_params["n_layer"] = max(4, min(current_layer + delta, 24))
            elif hints.get("n_layer"):
                layer_delta = int(round(2 * strength))
                new_params["n_layer"] = max(4, min(int(new_params["n_layer"]) + layer_delta, 24))
            if hints.get("n_layer") or "n_layer" in recs:
                print(f"[Agent 1] Adjusted layers: {new_params['n_layer']}")

            if "n_embd" in recs:
                current_embd = int(new_params.get("n_embd", 512))
                target_embd = max(128, int(round(float(recs["n_embd"]))))
                pull = min(0.8, 0.25 * strength)
                current_embd = int(current_embd * (1.0 - pull) + target_embd * pull)
                new_params["n_embd"] = min(max(128, current_embd), 1024)
            elif hints.get("n_embd"):
                emb_factor = 1.0 + 0.12 * strength
                new_params["n_embd"] = min(int(new_params["n_embd"] * emb_factor), 1024)
            if hints.get("n_embd") or "n_embd" in recs:
                print(f"[Agent 1] Adjusted embedding: {new_params['n_embd']}")
        else:
            # No summary yet - early iterations, try random exploration
            if iteration < 5:
                print("[Agent 1] Early iteration - random exploration")
                new_params["n_layer"] = random.randint(6, 18)
                new_params["n_embd"] = random.choice([256, 384, 512, 768, 1024])
                for lr_key in LR_KEYS:
                    new_params[lr_key] = self._random_lr(lr_key)

        for key in LR_KEYS:
            new_params[key] = self._clamp_lr(key, float(new_params.get(key, LR_DEFAULTS[key])))

        return new_params

    def _clamp_lr(self, key: str, lr: float) -> float:
        """Keep a learning-rate group inside its safe operating range."""
        lo, hi = self.lr_bounds.get(key, LR_SAFE_RANGES.get(key, (1e-5, 5.0)))
        if not math.isfinite(lr):
            return lo
        return max(lo, min(lr, hi))

    def _random_lr(self, key: str) -> float:
        """Log-uniform random sample within the safe range for one LR group."""
        lo, hi = self.lr_bounds.get(key, LR_SAFE_RANGES[key])
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))

    def _importance_magnitude(self, score: float) -> float:
        """Map [0,1] score to distance from neutral 0.5 in [0,1]."""
        bounded = max(0.0, min(1.0, float(score)))
        return abs(bounded - 0.5) * 2.0

    def _summary_importance_hints(self, summary: str) -> Dict[str, bool]:
        """Extract lightweight importance hints from summary text."""
        if not summary:
            return {}
        text = summary.lower()
        important_tokens = ["important", "stable", "strong", "matters"]

        def _has_signal(*tokens: str) -> bool:
            return any(token in text for token in tokens) and any(t in text for t in important_tokens)

        return {
            "embedding_lr": _has_signal("embedding_lr", "embedding lr"),
            "unembedding_lr": _has_signal("unembedding_lr", "unembedding lr"),
            "matrix_lr": _has_signal("matrix_lr", "matrix lr", "learning rate", "learning_rate", "lr"),
            "scalar_lr": _has_signal("scalar_lr", "scalar lr"),
            "n_layer": _has_signal("n_layer", "layer", "depth"),
            "n_embd": _has_signal("n_embd", "embedding"),
            "n_head": _has_signal("n_head", "head"),
        }

    def _extract_summary_recommendations(self, summary: str) -> Dict[str, float]:
        """Parse numeric recommendation lines from Agent 3 markdown summary."""
        if not summary:
            return {}

        patterns = {
            # Negative lookbehind on "embedding_lr" keeps it from also matching
            # inside "unembedding_lr" (which would otherwise double-match).
            "embedding_lr": r"(?<!un)embedding_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "unembedding_lr": r"unembedding_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "matrix_lr": r"matrix_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "scalar_lr": r"scalar_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "n_layer": r"n_layer[^\n:]*:\s*([0-9eE+\-.]+)",
            "n_embd": r"n_embd[^\n:]*:\s*([0-9eE+\-.]+)",
            "n_head": r"n_head[^\n:]*:\s*([0-9eE+\-.]+)",
        }

        recommendations: Dict[str, float] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, summary, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                recommendations[key] = float(match.group(1))
            except (TypeError, ValueError):
                continue
        return recommendations

    def train_model(
        self, hyperparams: Dict[str, Any], dry_run: bool = False, iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Train model and return metrics.

        Uses a real subprocess when available, but gracefully falls back to a
        lightweight simulated run so the multi-agent loop still produces
        meaningful artifacts in local or constrained environments.
        """
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

        # --- Priority 1: remote GPU server (configured via .env) ---
        try:
            from agents.remote_runner import is_remote_configured, run_training_remote
            if is_remote_configured():
                print("[Agent 1] Remote GPU server configured — running training remotely")
                metrics = run_training_remote(
                    hyperparams_local_path=str(self.model_config_path),
                    timeout=self.training_budget + 60,
                )
                print(f"[Agent 1] Remote training complete. Metrics: {metrics}")
                return metrics
        except Exception as e:
            print(f"[Agent 1] Remote training failed ({e}) — falling back to local")

        # --- Priority 2: local uv/train.py subprocess ---
        try:
            if not self._can_run_training_command():
                raise RuntimeError("training command unavailable")

            result = subprocess.run(
                ["uv", "run", "train.py"],
                capture_output=True,
                text=True,
                timeout=self.training_budget + 30,
            )
            metrics = self._parse_training_output(result.stdout)
            print(f"[Agent 1] Training complete. Metrics: {metrics}")
            metrics.setdefault("status", "ok")
            return metrics

        except subprocess.TimeoutExpired:
            print("[Agent 1] Training timeout")
            return {"val_bpb": float("inf"), "error": "timeout", "status": "simulated"}
        except Exception as e:
            print(f"[Agent 1] Training error: {e}")
            return self._simulate_training_result(hyperparams, iteration, str(e))

    def _can_run_training_command(self) -> bool:
        """Return True when the training subprocess can be executed."""
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _simulate_training_result(
        self, hyperparams: Dict[str, Any], iteration: int, error: str
    ) -> Dict[str, Any]:
        """Generate a deterministic surrogate metric for local testing."""
        matrix_lr = float(hyperparams.get("matrix_lr", LR_DEFAULTS["matrix_lr"]))
        depth = int(hyperparams.get("n_layer", 12))
        width = int(hyperparams.get("n_embd", 512))
        base = 1.25 - (0.002 * min(depth, 20)) - (0.000001 * width) + (0.15 * min(matrix_lr / LR_DEFAULTS["matrix_lr"], 2.0))
        val_bpb = max(0.65, base - 0.001 * iteration)
        return {
            "val_bpb": round(val_bpb, 6),
            "training_time": round(0.2 + 0.01 * iteration, 3),
            "checkpoint_path": None,
            "status": "simulated",
            "error": error,
        }

    _TRAIN_OUTPUT_FIELDS = {
        "val_bpb:": ("val_bpb", float),
        "training_seconds:": ("training_time", float),
        "total_seconds:": ("total_seconds", float),
        "peak_vram_mb:": ("peak_vram_mb", float),
        "mfu_percent:": ("mfu_percent", float),
        "total_tokens_m:": ("total_tokens_M", float),
        "num_steps:": ("num_steps", int),
        "num_params_m:": ("num_params_M", float),
        "depth:": ("depth", int),
    }

    def _parse_training_output(self, stdout: str) -> Dict[str, Any]:
        """Parse all metrics from train.py's final summary block."""
        metrics: Dict[str, Any] = {"val_bpb": float("inf")}
        for line in stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[0].lower()
            if key in self._TRAIN_OUTPUT_FIELDS and len(parts) >= 2:
                dest, cast = self._TRAIN_OUTPUT_FIELDS[key]
                try:
                    metrics[dest] = cast(parts[1])
                except (ValueError, IndexError):
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
Example: {{"n_layer": 14, "matrix_lr": 0.03}}"""

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
