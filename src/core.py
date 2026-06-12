"""Core shared types and utilities for PCMOBO."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PORTFOLIO_ACQ_NAMES = ("qlogehvi", "qlognparego", "qucb", "qlogpof")
ALL_ACQ_NAMES = PORTFOLIO_ACQ_NAMES
FIXED_POLICY_NAMES = PORTFOLIO_ACQ_NAMES
DYNAMIC_POLICY_NAMES = ("llm_switch", "bandit_ucb_switch")
POLICY_NAMES = (*FIXED_POLICY_NAMES, *DYNAMIC_POLICY_NAMES)

ACQ_ALIASES = {
    "ehvi": "qlogehvi",
    "qehvi": "qlogehvi",
    "nparego": "qlognparego",
    "ucb": "qucb",
    "pof": "qlogpof",
}
POLICY_ALIASES = {
    **ACQ_ALIASES,
    "bandit_ucb": "bandit_ucb_switch",
    "bandit": "bandit_ucb_switch",
}


@dataclass(frozen=True)
class DeviceConfig:
    device: str = "auto"
    dtype: str = "float64"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dim: int
    num_objectives: int
    num_constraints: int


@dataclass
class IterationRecord:
    iteration: int
    selected_acq: str
    selected_source: str
    feasible_observed: int
    cumulative_feasible_count: int
    hypervolume: float
    reason: str
    reflection: str
    confidence: float
    memory_snapshot: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Iteration": int(self.iteration),
            "SelectedAcq": str(self.selected_acq),
            "DecisionSource": str(self.selected_source),
            "ObservedFeasible": int(self.feasible_observed),
            "CumulativeFeasibleCount": int(self.cumulative_feasible_count),
            "Hypervolume": float(self.hypervolume),
            "Reason": str(self.reason),
            "Reflection": str(self.reflection),
            "Confidence": float(self.confidence),
            "MemorySnapshotBeforeDecision": str(self.memory_snapshot),
        }


def parse_seed_list(spec: str) -> List[int]:
    seeds: List[int] = []
    for token in spec.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            seeds.append(int(part))
    return seeds


def normalize_acq_name(name: str) -> str:
    key = str(name).strip().lower()
    return ACQ_ALIASES.get(key, key)


def normalize_policy_name(name: str) -> str:
    key = str(name).strip().lower()
    return POLICY_ALIASES.get(key, key)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compact_json(data: Any, max_chars: int = 3000) -> str:
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
        return default
    except Exception:
        return default


def ci95(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), 0.0
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / math.sqrt(n))
    return mean, float(1.96 * sem)
