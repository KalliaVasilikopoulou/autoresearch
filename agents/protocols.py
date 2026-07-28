from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TrainingResult:
    run_id: str
    hyperparams: Dict[str, Any]
    val_bpb: float
    training_time: float
    checkpoint_path: Optional[str] = None
    report_id: Optional[str] = None
    status: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisEvidence:
    report_id: str
    model_id: str
    important_heads: List[Dict[str, Any]]
    hyperparameter_importance: Dict[str, float]
    stuck_signal: bool
    confidence: float
    notes: List[str] = field(default_factory=list)
    token_fingerprint: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryEvidence:
    summary_id: str
    batch_size: int
    stable_patterns: List[str] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    recommended_hyperparams: Dict[str, Any] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
