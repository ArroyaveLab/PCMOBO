"""UCB bandit over acquisition portfolio arms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from core import PORTFOLIO_ACQ_NAMES, safe_float


@dataclass
class BanditUCBSwitchPolicy:
    arms: tuple[str, ...] = PORTFOLIO_ACQ_NAMES
    exploration_c: float = 1.0
    weight_feasible: float = 0.7
    weight_hv: float = 0.3
    warmstart: bool = True
    counts: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    positive_hv_history: List[float] = field(default_factory=list)
    last_ucb: Dict[str, float] = field(default_factory=dict)
    last_warmstart_flag: int = 0

    def __post_init__(self) -> None:
        self.arms = tuple(str(a).strip().lower() for a in self.arms)
        if not self.arms:
            raise ValueError("Bandit policy requires at least one arm.")
        if not np.isclose(self.weight_feasible + self.weight_hv, 1.0, atol=1e-9):
            total = max(1e-9, self.weight_feasible + self.weight_hv)
            self.weight_feasible /= total
            self.weight_hv /= total
        for arm in self.arms:
            self.counts[arm] = 0
            self.means[arm] = 0.0
            self.last_ucb[arm] = 0.0

    def _hv_scale(self) -> float:
        if not self.positive_hv_history:
            return 1.0
        return max(1e-9, float(np.median(np.array(self.positive_hv_history, dtype=float))))

    def select(
        self, iteration: int, acq_diagnostics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        t = max(1, int(iteration))
        self.last_warmstart_flag = 0

        if self.warmstart:
            for arm in self.arms:
                if self.counts[arm] == 0:
                    self.last_warmstart_flag = 1
                    self.last_ucb = {a: float("nan") for a in self.arms}
                    return {
                        "acq": arm,
                        "reason": f"bandit_warmstart_arm={arm}",
                        "reflection": "ucb_warmstart_round_robin",
                        "confidence": 0.6,
                        "source": "bandit_ucb",
                        "prompt_version": "none",
                    }

        best_arm = self.arms[0]
        best_score = float("-inf")
        ucb_scores: Dict[str, float] = {}
        for arm in self.arms:
            n = max(1, int(self.counts[arm]))
            bonus = float(self.exploration_c) * math.sqrt(math.log(t + 1.0) / n)
            score = float(self.means[arm] + bonus)
            ucb_scores[arm] = score
            if score > best_score:
                best_score = score
                best_arm = arm
            elif score == best_score:
                # Deterministic tie-break with predicted feasibility, then objective score.
                best_diag = acq_diagnostics.get(best_arm, {})
                curr_diag = acq_diagnostics.get(arm, {})
                curr_key = (
                    safe_float(curr_diag.get("pred_feas_prob", 0.0)),
                    safe_float(curr_diag.get("pred_obj_score", 0.0)),
                )
                best_key = (
                    safe_float(best_diag.get("pred_feas_prob", 0.0)),
                    safe_float(best_diag.get("pred_obj_score", 0.0)),
                )
                if curr_key > best_key:
                    best_arm = arm
        self.last_ucb = ucb_scores
        return {
            "acq": best_arm,
            "reason": f"bandit_ucb_select_arm={best_arm}",
            "reflection": "ucb_exploit_explore_tradeoff",
            "confidence": 0.7,
            "source": "bandit_ucb",
            "prompt_version": "none",
        }

    def update(self, chosen_arm: str, feasible_gain: int, hv_gain: float) -> Dict[str, float]:
        arm = str(chosen_arm).strip().lower()
        if arm not in self.counts:
            return {
                "reward": 0.0,
                "reward_feasible_term": 0.0,
                "reward_hv_term": 0.0,
                "hv_scale": self._hv_scale(),
            }
        hv_scale = self._hv_scale()
        feasible_term = 1.0 if int(feasible_gain) > 0 else 0.0
        hv_term = min(max(float(hv_gain), 0.0) / hv_scale, 1.0)
        reward = float(self.weight_feasible * feasible_term + self.weight_hv * hv_term)

        self.counts[arm] += 1
        n = self.counts[arm]
        old = self.means[arm]
        self.means[arm] = old + (reward - old) / float(n)
        if float(hv_gain) > 0.0:
            self.positive_hv_history.append(float(hv_gain))
        return {
            "reward": reward,
            "reward_feasible_term": feasible_term,
            "reward_hv_term": hv_term,
            "hv_scale": hv_scale,
        }
