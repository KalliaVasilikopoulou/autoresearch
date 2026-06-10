"""Fast XAI methods: Ablation and Partial Dependence analysis."""

import torch
from typing import Dict, Tuple, List
import numpy as np


class FastXAIMethods:
    """Fast, practical XAI techniques for Agent 2."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def top_k_ablation_study(
        self,
        model,
        val_dataloader,
        evaluate_fn,
        k: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Ablate top-K attention heads by weight magnitude.

        Returns:
            {head_description: importance_score}
        """
        # Get baseline performance
        baseline_bpb = evaluate_fn(model, val_dataloader)

        impacts = {}
        head_count = 0

        # Identify top-K heads by weight magnitude
        for layer_idx, layer in enumerate(model.transformer.h):
            # Attention weights: (n_head, 3*n_embd)
            attn_weights = layer.attn.c_attn.weight  # Shape: (3*n_embd, n_embd)
            n_head = model.config.n_head
            head_dim = model.config.n_embd // n_head

            # Per-head magnitude
            head_magnitudes = []
            for head_idx in range(n_head):
                # Extract weights for this head
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                head_weight = attn_weights[:, start:end]
                magnitude = torch.norm(head_weight).item()
                head_magnitudes.append((head_idx, magnitude))

            # Sort by magnitude, keep top-K
            head_magnitudes.sort(key=lambda x: x[1], reverse=True)

            for head_idx, magnitude in head_magnitudes[:k]:
                # Temporarily disable this head
                original_scale = layer.attn.head_scales[head_idx].clone()
                layer.attn.head_scales[head_idx] *= 0.0  # Ablate

                # Evaluate
                ablated_bpb = evaluate_fn(model, val_dataloader)

                # Restore
                layer.attn.head_scales[head_idx] = original_scale

                # Record impact
                impact = baseline_bpb - ablated_bpb  # Positive = important
                head_key = f"L{layer_idx}_H{head_idx}"
                impacts[head_key] = impact

                head_count += 1
                if verbose and head_count % 5 == 0:
                    print(f"Ablated {head_count} heads...")

        return impacts

    def partial_dependence_hyperparams(
        self,
        model,
        val_dataloader,
        evaluate_fn,
        hyperparams_used: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float, int]] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Vary key hyperparameters, measure effect on validation metric.

        param_ranges format:
            {"learning_rate": (1e-5, 1e-3, 5)}  # min, max, num_points
        """
        if param_ranges is None:
            param_ranges = {
                "n_layers": (1, min(hyperparams_used.get("n_layers", 12) + 4, 24), 3),
                "n_embd": (256, 1024, 3),
            }

        results = {}

        for param_name, (param_min, param_max, num_points) in param_ranges.items():
            if param_name not in hyperparams_used:
                continue

            values = np.linspace(param_min, param_max, num_points)
            curve = []

            for value in values:
                # Note: This is a MOCK - actual implementation would retrain
                # For now, we estimate based on current model
                # In practice, this would involve actual model training with varied params
                estimated_bpb = self._estimate_metric_for_param(
                    param_name, value, hyperparams_used
                )
                curve.append((value, estimated_bpb))

            results[param_name] = curve

        return results

    def _estimate_metric_for_param(
        self, param_name: str, param_value: float, baseline_params: Dict[str, float]
    ) -> float:
        """
        Heuristic estimation of metric for parameter variation.
        (Placeholder - in production would do actual training)
        """
        # These are mock heuristics
        baseline_bpb = 1.0

        if param_name == "learning_rate":
            # Learning rate has sweet spot around 1e-3
            optimal = 1e-3
            distance = abs(np.log10(param_value) - np.log10(optimal))
            return baseline_bpb + distance * 0.1

        elif param_name == "n_layers":
            baseline_layers = baseline_params.get("n_layers", 12)
            if param_value == baseline_layers:
                return baseline_bpb
            elif param_value > baseline_layers:
                # More layers helps until diminishing returns
                return baseline_bpb - (param_value - baseline_layers) * 0.01
            else:
                return baseline_bpb + (baseline_layers - param_value) * 0.02

        elif param_name == "n_embd":
            baseline_embd = baseline_params.get("n_embd", 512)
            if param_value == baseline_embd:
                return baseline_bpb
            # Larger embedding helps
            return baseline_bpb - np.log2(param_value / baseline_embd) * 0.05

        return baseline_bpb

    def detect_stuck_signal(
        self, all_impacts: List[Dict[str, float]], threshold: int = 5
    ) -> bool:
        """
        Detect if model is stuck (same pattern as N previous models).
        """
        if len(all_impacts) < threshold:
            return False

        # Compare top-5 important heads across last N models
        recent_tops = []
        for impacts_dict in all_impacts[-threshold:]:
            if not impacts_dict:
                continue
            top_5 = sorted(impacts_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            recent_tops.append([head for head, _ in top_5])

        # Check if all recent models have similar top heads
        if not recent_tops:
            return False

        first_top = set(recent_tops[0])
        all_similar = all(set(tops) == first_top for tops in recent_tops)

        return all_similar


class ThoroughXAIMethods:
    """Thorough but slower XAI methods (for future use)."""

    def circuits_analysis(self, model, input_ids: torch.Tensor) -> Dict[str, List[str]]:
        """
        Trace attention circuits through the model.
        (Placeholder for future implementation)
        """
        # This would trace information flow through layers
        # Requires more sophisticated analysis
        pass

    def svcca_analysis(
        self, model1_activations, model2_activations
    ) -> Dict[int, float]:
        """
        Compare internal representations using SVCCA.
        (Placeholder for future implementation)
        """
        # Singular Vector Canonical Correlation Analysis
        # Requires activation recording from two models
        pass
