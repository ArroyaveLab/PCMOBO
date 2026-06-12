"""Single-seed constrained MO BO loop with acquisition-portfolio switching."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from acq import AcqState, propose_candidate
from benchmarks import build_benchmark
from core import (
    ALL_ACQ_NAMES,
    FIXED_POLICY_NAMES,
    PORTFOLIO_ACQ_NAMES,
    IterationRecord,
    compact_json,
    safe_float,
)
from engine.modeling import compute_hv, fit_model_bundle
from paths import SYNTHETIC_RUNS_ROOT
from policy import BanditUCBSwitchPolicy, LLMSwitchPolicy


@dataclass
class SeedRunConfig:
    benchmark: str
    policy: str
    seed: int
    iterations: int = 100
    q: int = 1
    n_init: int = 0
    results_dir: str = str(SYNTHETIC_RUNS_ROOT / "_tmp")
    llm_model: str = "gpt-4o"
    llm_provider: str = "openai"
    llm_timeout: int = 90
    memory_window: int = 10
    env_path: str = ".env"
    device: str = "auto"
    dtype: str = "float64"
    acq_mc_samples: int = 128
    acq_raw_samples: int = 256
    acq_num_restarts: int = 8
    acq_beta: float = 0.2
    bandit_c: float = 1.0
    bandit_w_feas: float = 0.7
    bandit_w_hv: float = 0.3
    bandit_warmstart: int = 1


def _resolve_torch(cfg: SeedRunConfig) -> tuple[Any, Any]:
    import torch

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    return device, dtype


def _full_history_aggregates(
    events: List[Dict[str, Any]],
    acq_names: tuple[str, ...] = ALL_ACQ_NAMES,
) -> Dict[str, Any]:
    if not events:
        return {
            "n_events": 0,
            "mean_feasible_gain": 0.0,
            "mean_hv_gain": 0.0,
            "by_acq": {},
        }

    out = {
        "n_events": int(len(events)),
        "mean_feasible_gain": float(
            np.mean([safe_float(e.get("feasible_gain", 0.0)) for e in events])
        ),
        "mean_hv_gain": float(np.mean([safe_float(e.get("hv_gain", 0.0)) for e in events])),
        "by_acq": {},
    }
    for acq in acq_names:
        subset = [e for e in events if str(e.get("selected_acq", "")).lower() == acq]
        if not subset:
            out["by_acq"][acq] = {
                "count": 0,
                "feasible_gain_rate": 0.0,
                "mean_hv_gain": 0.0,
            }
            continue
        feas_rate = float(
            np.mean([1.0 if safe_float(e.get("feasible_gain", 0.0)) > 0 else 0.0 for e in subset])
        )
        mean_hv_gain = float(np.mean([safe_float(e.get("hv_gain", 0.0)) for e in subset]))
        out["by_acq"][acq] = {
            "count": int(len(subset)),
            "feasible_gain_rate": feas_rate,
            "mean_hv_gain": mean_hv_gain,
        }
    return out


def _recent_stats(events: List[Dict[str, Any]], window: int) -> Dict[str, float]:
    if not events:
        return {
            "recent_window": 0,
            "recent_feasible_gain_rate": 0.0,
            "recent_hv_gain_mean": 0.0,
            "recent_hv_gain_std": 0.0,
        }
    tail = events[-max(1, int(window)) :]
    hv_gains = [safe_float(e.get("hv_gain", 0.0)) for e in tail]
    return {
        "recent_window": int(len(tail)),
        "recent_feasible_gain_rate": float(
            np.mean([1.0 if safe_float(e.get("feasible_gain", 0.0)) > 0 else 0.0 for e in tail])
        ),
        "recent_hv_gain_mean": float(np.mean(hv_gains)),
        "recent_hv_gain_std": float(np.std(hv_gains)),
    }


def _feasibility_stage(
    cumulative_feasible_count: int,
    observed_feasible_fraction: float,
    recent_feasible_gain_rate: float,
) -> str:
    if int(cumulative_feasible_count) <= 0:
        return "none"
    if (
        safe_float(observed_feasible_fraction, 0.0) < 0.10
        or safe_float(recent_feasible_gain_rate, 0.0) < 0.20
    ):
        return "fragile"
    return "stable"


def _normalize_metric_map(values: Dict[str, float], default: float = 0.5) -> Dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {k: float(default) for k in values}
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return {k: float(default) for k in values}
    out: Dict[str, float] = {}
    for key, value in values.items():
        v = float(value)
        if not math.isfinite(v):
            out[key] = 0.0
        else:
            out[key] = float((v - lo) / (hi - lo))
    return out


def _rank_metric_map(values: Dict[str, float]) -> Dict[str, int]:
    def _sort_key(item: tuple[str, float]) -> tuple[int, float, str]:
        name, value = item
        v = float(value)
        if not math.isfinite(v):
            return (1, 0.0, name)
        return (0, -v, name)

    return {name: idx + 1 for idx, (name, _) in enumerate(sorted(values.items(), key=_sort_key))}


def _last_selected_acq(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    return str(events[-1].get("selected_acq", "")).strip().lower()


def _same_acq_streak(events: List[Dict[str, Any]]) -> int:
    last = _last_selected_acq(events)
    if not last:
        return 0
    streak = 0
    for event in reversed(events):
        if str(event.get("selected_acq", "")).strip().lower() != last:
            break
        streak += 1
    return int(streak)


def _iterations_since_last_switch(events: List[Dict[str, Any]]) -> int:
    return max(0, _same_acq_streak(events) - 1)


def _current_arm_no_hv_gain_streak(events: List[Dict[str, Any]]) -> int:
    last = _last_selected_acq(events)
    if not last:
        return 0
    same_arm_streak = _same_acq_streak(events)
    if same_arm_streak <= 1:
        return 0
    streak = 0
    for event in reversed(events):
        if str(event.get("selected_acq", "")).strip().lower() != last:
            break
        if safe_float(event.get("hv_gain", 0.0), 0.0) > 0.0:
            break
        streak += 1
    return int(min(streak, same_arm_streak - 1))


def _iterations_since_selected(events: List[Dict[str, Any]], acq: str, iteration: int) -> int:
    target = str(acq).strip().lower()
    current_index = max(0, int(iteration) - 1)
    for event in reversed(events):
        if str(event.get("selected_acq", "")).strip().lower() == target:
            return max(0, current_index - int(event.get("iteration", current_index)))
    return int(current_index)


def _recent_arm_stats(events: List[Dict[str, Any]], acq: str, window: int) -> Dict[str, float]:
    target = str(acq).strip().lower()
    tail = events[-max(1, int(window)) :] if events else []
    subset = [e for e in tail if str(e.get("selected_acq", "")).strip().lower() == target]
    if not subset:
        return {
            "recent_count": 0,
            "recent_feasible_gain_rate": 0.0,
            "recent_hv_gain_mean": 0.0,
        }
    return {
        "recent_count": int(len(subset)),
        "recent_feasible_gain_rate": float(
            np.mean([1.0 if safe_float(e.get("feasible_gain", 0.0)) > 0 else 0.0 for e in subset])
        ),
        "recent_hv_gain_mean": float(np.mean([safe_float(e.get("hv_gain", 0.0)) for e in subset])),
    }


def _best_metric_summary(per_arm: Dict[str, Dict[str, Any]], key: str) -> tuple[str, float]:
    best_arm = ""
    best_value = float("nan")
    for arm in PORTFOLIO_ACQ_NAMES:
        value = safe_float(per_arm.get(arm, {}).get(key, float("nan")), float("nan"))
        if not math.isfinite(value):
            continue
        if (
            not best_arm
            or value > best_value
            or (abs(value - best_value) < 1e-12 and arm < best_arm)
        ):
            best_arm = arm
            best_value = float(value)
    return best_arm, float(best_value)


def _safe_gap(value: float, best_value: float) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(best_value)):
        return float("nan")
    return float(value - best_value)


def _memory_item_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iteration": int(event.get("iteration", 0)),
        "chosen_acq": str(event.get("chosen_acq", event.get("selected_acq", ""))),
        "observed_feasible": int(event.get("observed_feasible", 0)),
        "feasible_gain": int(event.get("feasible_gain", 0)),
        "hv_gain": float(safe_float(event.get("hv_gain", 0.0))),
        "cumulative_feasible_count": int(event.get("cumulative_feasible_count", 0)),
        "feasibility_stage_at_decision": str(event.get("feasibility_stage_at_decision", "none")),
        "same_acq_streak_at_decision": int(event.get("same_acq_streak_at_decision", 0) or 0),
        "current_arm_no_hv_gain_streak_at_decision": int(
            event.get("current_arm_no_hv_gain_streak_at_decision", 0) or 0
        ),
        "chosen_predicted_feasibility": float(
            safe_float(event.get("chosen_predicted_feasibility", float("nan")), float("nan"))
        ),
        "chosen_hv_proxy": float(
            safe_float(event.get("chosen_hv_proxy", float("nan")), float("nan"))
        ),
        "chosen_uncertainty": float(
            safe_float(event.get("chosen_uncertainty", float("nan")), float("nan"))
        ),
        "chosen_novelty": float(
            safe_float(event.get("chosen_novelty", float("nan")), float("nan"))
        ),
        "best_feas_arm": str(event.get("best_feas_arm", "")),
        "best_feas_value": float(
            safe_float(event.get("best_feas_value", float("nan")), float("nan"))
        ),
        "best_hv_arm": str(event.get("best_hv_arm", "")),
        "best_hv_value": float(safe_float(event.get("best_hv_value", float("nan")), float("nan"))),
        "best_uncertainty_arm": str(event.get("best_uncertainty_arm", "")),
        "best_uncertainty_value": float(
            safe_float(event.get("best_uncertainty_value", float("nan")), float("nan"))
        ),
        "chosen_vs_best_feas_gap": float(
            safe_float(event.get("chosen_vs_best_feas_gap", float("nan")), float("nan"))
        ),
        "chosen_vs_best_hv_gap": float(
            safe_float(event.get("chosen_vs_best_hv_gap", float("nan")), float("nan"))
        ),
    }


def _build_portfolio_context(
    acq_diagnostics: Dict[str, Dict[str, float]],
    events: List[Dict[str, Any]],
    iteration: int,
    window: int,
) -> Dict[str, Any]:
    acq_values = {
        arm: safe_float(acq_diagnostics.get(arm, {}).get("acq_value", float("nan")), float("nan"))
        for arm in PORTFOLIO_ACQ_NAMES
    }
    pred_feas = {
        arm: safe_float(
            acq_diagnostics.get(arm, {}).get("pred_feas_prob", float("nan")), float("nan")
        )
        for arm in PORTFOLIO_ACQ_NAMES
    }
    hv_proxy = {
        arm: safe_float(
            acq_diagnostics.get(arm, {}).get("posterior_mean_hv_proxy", float("nan")),
            float("nan"),
        )
        for arm in PORTFOLIO_ACQ_NAMES
    }
    uncertainty = {
        arm: safe_float(
            acq_diagnostics.get(arm, {}).get("candidate_uncertainty", float("nan")),
            float("nan"),
        )
        for arm in PORTFOLIO_ACQ_NAMES
    }
    novelty = {
        arm: safe_float(
            acq_diagnostics.get(arm, {}).get("candidate_novelty", float("nan")),
            float("nan"),
        )
        for arm in PORTFOLIO_ACQ_NAMES
    }

    norm_acq = _normalize_metric_map(acq_values, default=0.5)
    feas_rank = _rank_metric_map(pred_feas)
    hv_rank = _rank_metric_map(hv_proxy)
    uncertainty_rank = _rank_metric_map(uncertainty)

    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in PORTFOLIO_ACQ_NAMES:
        recent = _recent_arm_stats(events, arm, window)
        per_arm[arm] = {
            "predicted_feasibility": float(pred_feas[arm]),
            "pred_obj_score": float(
                safe_float(
                    acq_diagnostics.get(arm, {}).get("pred_obj_score", float("nan")), float("nan")
                )
            ),
            "normalized_acq_value": float(norm_acq[arm]),
            "posterior_mean_hv_proxy": float(hv_proxy[arm]),
            "candidate_uncertainty": float(uncertainty[arm]),
            "candidate_novelty": float(novelty[arm]),
            "recent_count": int(recent["recent_count"]),
            "recent_feasible_gain_rate": float(recent["recent_feasible_gain_rate"]),
            "recent_hv_gain_mean": float(recent["recent_hv_gain_mean"]),
            "iterations_since_selected": int(_iterations_since_selected(events, arm, iteration)),
            "feasibility_rank": int(feas_rank[arm]),
            "hv_proxy_rank": int(hv_rank[arm]),
            "uncertainty_rank": int(uncertainty_rank[arm]),
        }
    return {
        "per_arm": per_arm,
        "last_selected_acq": _last_selected_acq(events),
        "same_acq_streak": int(_same_acq_streak(events)),
        "iterations_since_last_switch": int(_iterations_since_last_switch(events)),
    }


def _build_llm_state(
    cfg: SeedRunConfig,
    acq_diagnostics: Dict[str, Dict[str, float]],
    events: List[Dict[str, Any]],
    iteration: int,
    current_hv: float,
    initial_hv: float,
    cumulative_feasible_count: int,
    num_observed: int,
) -> Dict[str, Any]:
    recent = _recent_stats(events, cfg.memory_window)
    observed_feasible_fraction = float(cumulative_feasible_count / max(1, num_observed))
    feasibility_stage = _feasibility_stage(
        cumulative_feasible_count=cumulative_feasible_count,
        observed_feasible_fraction=observed_feasible_fraction,
        recent_feasible_gain_rate=float(recent["recent_feasible_gain_rate"]),
    )
    portfolio_context = _build_portfolio_context(
        acq_diagnostics=acq_diagnostics,
        events=events,
        iteration=iteration,
        window=cfg.memory_window,
    )
    recent_history = [_memory_item_from_event(event) for event in events[-cfg.memory_window :]]

    return {
        "metadata": {
            "benchmark": cfg.benchmark,
            "iteration": int(iteration),
            "total_iterations": int(cfg.iterations),
            "allowed_arms": list(PORTFOLIO_ACQ_NAMES),
        },
        "global_state": {
            "progress_ratio": float(iteration / max(1, cfg.iterations)),
            "cumulative_feasible_count": int(cumulative_feasible_count),
            "current_hv": float(current_hv),
            "hv_gain_since_init": float(current_hv - initial_hv),
            "observed_feasible_fraction": float(observed_feasible_fraction),
            "recent_feasible_gain_rate": float(recent["recent_feasible_gain_rate"]),
            "recent_hv_gain_mean": float(recent["recent_hv_gain_mean"]),
            "recent_hv_gain_std": float(recent["recent_hv_gain_std"]),
            "feasibility_stage": feasibility_stage,
            "last_selected_acq": str(portfolio_context["last_selected_acq"]),
            "same_acq_streak": int(portfolio_context["same_acq_streak"]),
            "current_arm_no_hv_gain_streak": int(_current_arm_no_hv_gain_streak(events)),
            "iterations_since_last_switch": int(portfolio_context["iterations_since_last_switch"]),
        },
        "per_arm": portfolio_context["per_arm"],
        "recent_history": recent_history,
    }


def _decision_for_policy(
    cfg: SeedRunConfig,
    llm_policy: LLMSwitchPolicy | None,
    bandit_policy: BanditUCBSwitchPolicy | None,
    acq_diagnostics: Dict[str, Dict[str, float]],
    llm_state: Dict[str, Any],
) -> Dict[str, Any]:
    if cfg.policy in FIXED_POLICY_NAMES:
        return {
            "acq": cfg.policy,
            "reason": "fixed_baseline_policy",
            "reflection": "deterministic_fixed_acquisition",
            "confidence": 1.0,
            "source": "fixed_policy",
            "prompt_version": "none",
        }
    if cfg.policy == "bandit_ucb_switch":
        assert bandit_policy is not None
        return bandit_policy.select(
            iteration=int(llm_state.get("metadata", {}).get("iteration", 0)),
            acq_diagnostics=acq_diagnostics,
        )

    assert llm_policy is not None
    return llm_policy.decide(llm_state)


def run_single_seed(cfg: SeedRunConfig) -> int:
    import torch
    from torch.quasirandom import SobolEngine

    t0 = time.time()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device, dtype = _resolve_torch(cfg)
    adapter = build_benchmark(cfg.benchmark, device=device, dtype=dtype)
    bounds = adapter.bounds
    q = int(max(1, cfg.q))
    n_init = int(cfg.n_init) if cfg.n_init > 0 else max(10, 2 * adapter.dim)

    sobol = SobolEngine(dimension=adapter.dim, scramble=True, seed=cfg.seed)
    x_init_unit = sobol.draw(n_init).to(device=device, dtype=dtype)
    x_obs = bounds[0] + (bounds[1] - bounds[0]) * x_init_unit
    y_obj, y_con = adapter.evaluate(x_obs)

    llm_policy = None
    if cfg.policy == "llm_switch":
        llm_policy = LLMSwitchPolicy(
            model=cfg.llm_model,
            llm_provider=cfg.llm_provider,
            env_path=cfg.env_path,
            timeout_s=cfg.llm_timeout,
            memory_window=cfg.memory_window,
        )
    bandit_policy = None
    if cfg.policy == "bandit_ucb_switch":
        bandit_policy = BanditUCBSwitchPolicy(
            arms=PORTFOLIO_ACQ_NAMES,
            exploration_c=float(cfg.bandit_c),
            weight_feasible=float(cfg.bandit_w_feas),
            weight_hv=float(cfg.bandit_w_hv),
            warmstart=bool(int(cfg.bandit_warmstart)),
        )

    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    hv_prev = compute_hv(y_obj, y_con, adapter.reference_point)
    hv_init = float(hv_prev)
    feasible_prev = int((y_con <= 0.0).all(dim=-1).sum().item())

    bandit_stats = {
        "reward": float("nan"),
        "reward_feasible_term": float("nan"),
        "reward_hv_term": float("nan"),
        "hv_scale": float("nan"),
    }
    total_acq_failures = 0
    iterations_with_any_acq_failure = 0
    selected_proposal_failure_count = 0

    for it in range(1, int(cfg.iterations) + 1):
        bundle = fit_model_bundle(x_obs, y_obj, y_con)
        state = AcqState(
            x_obs=x_obs,
            y_obj=y_obj,
            y_con=y_con,
            bounds=bounds,
            reference_point=adapter.reference_point,
            q=q,
            mc_samples=int(cfg.acq_mc_samples),
            raw_samples=int(cfg.acq_raw_samples),
            num_restarts=int(cfg.acq_num_restarts),
            beta=float(cfg.acq_beta),
        )

        proposals: Dict[str, Dict[str, Any]] = {}
        for acq_name in ALL_ACQ_NAMES:
            proposals[acq_name] = propose_candidate(acq_name, bundle, state)
            proposals[acq_name]["status"] = "ok"
            proposals[acq_name]["error"] = ""

        acq_diags_all = {
            name: {
                "acq_value": safe_float(p.get("acq_value", float("nan")), float("nan")),
                "pred_feas_prob": safe_float(p.get("pred_feas_prob", float("nan")), float("nan")),
                "pred_obj_score": safe_float(p.get("pred_obj_score", float("nan")), float("nan")),
                "candidate_uncertainty": safe_float(
                    p.get("candidate_uncertainty", float("nan")),
                    float("nan"),
                ),
                "candidate_novelty": safe_float(
                    p.get("candidate_novelty", float("nan")),
                    float("nan"),
                ),
                "posterior_mean_hv_proxy": safe_float(
                    p.get("posterior_mean_hv_proxy", float("nan")),
                    float("nan"),
                ),
            }
            for name, p in proposals.items()
        }
        acq_diags_portfolio = {k: acq_diags_all[k] for k in PORTFOLIO_ACQ_NAMES}
        llm_state = _build_llm_state(
            cfg=cfg,
            acq_diagnostics=acq_diags_portfolio,
            events=events,
            iteration=it,
            current_hv=hv_prev,
            initial_hv=hv_init,
            cumulative_feasible_count=feasible_prev,
            num_observed=int(y_obj.shape[0]),
        )

        decision = _decision_for_policy(
            cfg=cfg,
            llm_policy=llm_policy,
            bandit_policy=bandit_policy,
            acq_diagnostics=acq_diags_portfolio,
            llm_state=llm_state,
        )
        chosen = str(decision.get("acq", "qlogpof")).strip().lower()
        if cfg.policy in FIXED_POLICY_NAMES:
            allowed = set(ALL_ACQ_NAMES)
        else:
            allowed = set(PORTFOLIO_ACQ_NAMES)
        if chosen not in allowed:
            raise ValueError(
                f"Invalid acquisition '{chosen}' from policy '{cfg.policy}' at seed={cfg.seed} "
                f"iteration={it}; allowed={sorted(allowed)}"
            )
            chosen = "qlogpof"
        num_acq_failures_this_iter = int(
            sum(
                1
                for proposal in proposals.values()
                if str(proposal.get("status", "unknown")) != "ok"
            )
        )
        any_acq_failure_this_iter = int(num_acq_failures_this_iter > 0)
        if any_acq_failure_this_iter:
            iterations_with_any_acq_failure += 1
            total_acq_failures += num_acq_failures_this_iter

        selected_proposal = proposals[chosen]
        selected_proposal_status = str(selected_proposal.get("status", "unknown"))
        selected_proposal_error = str(selected_proposal.get("error", ""))
        selected_proposal_failed = int(selected_proposal_status != "ok")
        selected_proposal_failure_count += selected_proposal_failed
        run_health_status = "degraded" if total_acq_failures > 0 else "valid"
        candidate = selected_proposal["candidate"]

        y_next_obj, y_next_con = adapter.evaluate(candidate)
        x_obs = torch.cat([x_obs, candidate], dim=0)
        y_obj = torch.cat([y_obj, y_next_obj], dim=0)
        y_con = torch.cat([y_con, y_next_con], dim=0)

        hv_now = compute_hv(y_obj, y_con, adapter.reference_point)
        feasible_now = int((y_con <= 0.0).all(dim=-1).sum().item())
        hv_gain = hv_now - hv_prev
        feasible_gain = feasible_now - feasible_prev
        feasible_observed = int((y_next_con <= 0.0).all(dim=-1).view(-1)[0].item())

        last_selected_acq = str(llm_state.get("global_state", {}).get("last_selected_acq", ""))
        same_acq_streak = int(llm_state.get("global_state", {}).get("same_acq_streak", 0))
        switch_occurred = int(bool(last_selected_acq) and chosen != last_selected_acq)
        pre_decision_per_arm = (
            llm_state.get("per_arm", {}) if isinstance(llm_state.get("per_arm"), dict) else {}
        )
        chosen_diag = acq_diags_all.get(chosen, {})
        best_feas_arm, best_feas_value = _best_metric_summary(
            pre_decision_per_arm, "predicted_feasibility"
        )
        best_hv_arm, best_hv_value = _best_metric_summary(
            pre_decision_per_arm, "posterior_mean_hv_proxy"
        )
        best_uncertainty_arm, best_uncertainty_value = _best_metric_summary(
            pre_decision_per_arm,
            "candidate_uncertainty",
        )
        chosen_predicted_feasibility = safe_float(
            chosen_diag.get("pred_feas_prob", float("nan")), float("nan")
        )
        chosen_hv_proxy = safe_float(
            chosen_diag.get("posterior_mean_hv_proxy", float("nan")), float("nan")
        )
        chosen_uncertainty = safe_float(
            chosen_diag.get("candidate_uncertainty", float("nan")), float("nan")
        )
        chosen_novelty = safe_float(
            chosen_diag.get("candidate_novelty", float("nan")), float("nan")
        )
        feasibility_stage = str(llm_state.get("global_state", {}).get("feasibility_stage", "none"))
        current_arm_no_hv_gain_streak = int(
            llm_state.get("global_state", {}).get("current_arm_no_hv_gain_streak", 0) or 0
        )

        event = {
            "iteration": it,
            "selected_acq": chosen,
            "chosen_acq": chosen,
            "decision_source": str(decision.get("source", "unknown")),
            "llm_pipeline_mode": str(decision.get("llm_pipeline_mode", "none")),
            "llm_model_requests_this_decision": int(
                decision.get("llm_model_requests_this_decision", 0) or 0
            ),
            "feasible_gain": int(feasible_gain),
            "hv_gain": float(hv_gain),
            "observed_feasible": int(feasible_observed),
            "cumulative_feasible_count": int(feasible_now),
            "hypervolume": float(hv_now),
            "feasibility_stage_at_decision": feasibility_stage,
            "same_acq_streak_at_decision": int(same_acq_streak),
            "current_arm_no_hv_gain_streak_at_decision": int(current_arm_no_hv_gain_streak),
            "chosen_predicted_feasibility": float(chosen_predicted_feasibility),
            "chosen_hv_proxy": float(chosen_hv_proxy),
            "chosen_uncertainty": float(chosen_uncertainty),
            "chosen_novelty": float(chosen_novelty),
            "best_feas_arm": best_feas_arm,
            "best_feas_value": float(best_feas_value),
            "best_hv_arm": best_hv_arm,
            "best_hv_value": float(best_hv_value),
            "best_uncertainty_arm": best_uncertainty_arm,
            "best_uncertainty_value": float(best_uncertainty_value),
            "chosen_vs_best_feas_gap": float(
                _safe_gap(chosen_predicted_feasibility, best_feas_value)
            ),
            "chosen_vs_best_hv_gap": float(_safe_gap(chosen_hv_proxy, best_hv_value)),
            "last_selected_acq": last_selected_acq,
            "same_acq_streak": int(same_acq_streak),
            "switch_occurred": int(switch_occurred),
            "any_acq_failure_this_iter": int(any_acq_failure_this_iter),
            "num_acq_failures_this_iter": int(num_acq_failures_this_iter),
            "selected_proposal_failed": int(selected_proposal_failed),
            "selected_proposal_status": selected_proposal_status,
            "selected_proposal_error": selected_proposal_error,
            "total_acq_failures": int(total_acq_failures),
            "iterations_with_any_acq_failure": int(iterations_with_any_acq_failure),
            "selected_proposal_failure_count": int(selected_proposal_failure_count),
            "run_health_status": run_health_status,
        }
        if bandit_policy is not None:
            bandit_stats = bandit_policy.update(
                chosen_arm=chosen,
                feasible_gain=feasible_gain,
                hv_gain=hv_gain,
            )
            event["bandit_reward"] = float(bandit_stats["reward"])
            event["bandit_reward_feasible_term"] = float(bandit_stats["reward_feasible_term"])
            event["bandit_reward_hv_term"] = float(bandit_stats["reward_hv_term"])
            event["bandit_hv_scale"] = float(bandit_stats["hv_scale"])
        events.append(event)

        memory_snapshot = compact_json(llm_state, max_chars=6000)
        rec = IterationRecord(
            iteration=it,
            selected_acq=chosen,
            selected_source=str(decision.get("source", "unknown")),
            feasible_observed=feasible_observed,
            cumulative_feasible_count=feasible_now,
            hypervolume=hv_now,
            reason=str(decision.get("reason", "")),
            reflection=str(decision.get("reflection", "")),
            confidence=float(np.clip(safe_float(decision.get("confidence", 0.5), 0.5), 0.0, 1.0)),
            memory_snapshot=memory_snapshot,
        ).as_dict()
        rec.update(
            {
                "Benchmark": adapter.name,
                "Policy": cfg.policy,
                "Seed": int(cfg.seed),
                "q": int(cfg.q),
                "Dim": int(adapter.dim),
                "NumObjectives": int(adapter.num_objectives),
                "NumConstraints": int(adapter.num_constraints),
                "DecisionPromptVersion": str(decision.get("prompt_version", "none")),
                "LLMPipelineMode": str(decision.get("llm_pipeline_mode", "none")),
                "LLMModelRequestsThisDecision": int(
                    decision.get("llm_model_requests_this_decision", 0) or 0
                ),
                "LLMFeasibilityAdvocateJSON": str(
                    decision.get("llm_feasibility_advocate_json", "")
                ),
                "LLMHVAdvocateJSON": str(decision.get("llm_hv_advocate_json", "")),
                "LLMArbiterJSON": str(decision.get("llm_arbiter_json", "")),
                "LLMStageError": str(decision.get("llm_stage_error", "")),
                "XChosen": json.dumps(candidate.detach().cpu().view(-1).tolist()),
                "YObserved": json.dumps(y_next_obj.detach().cpu().view(-1).tolist()),
                "CObserved": json.dumps(y_next_con.detach().cpu().view(-1).tolist()),
                "HVGain": float(hv_gain),
                "FeasibleGain": int(feasible_gain),
                "LLMDecisionCount": int(
                    sum(1 for e in events if e["decision_source"] == "llm_three_agent")
                ),
                "LLMModelRequestCount": int(
                    sum(int(e.get("llm_model_requests_this_decision", 0) or 0) for e in events)
                ),
                "RunElapsedSec": float(time.time() - t0),
                "LastSelectedAcq": last_selected_acq,
                "SameAcqStreak": int(same_acq_streak),
                "SwitchOccurred": int(switch_occurred),
                "FeasibilityStage": feasibility_stage,
                "CurrentArmNoHVGainStreak": int(current_arm_no_hv_gain_streak),
                "ObservedFeasibleFraction": float(
                    safe_float(
                        llm_state.get("global_state", {}).get("observed_feasible_fraction", 0.0)
                    )
                ),
                "RecentFeasibleGainRate": float(
                    safe_float(
                        llm_state.get("global_state", {}).get("recent_feasible_gain_rate", 0.0)
                    )
                ),
                "RecentHVGainMean": float(
                    safe_float(llm_state.get("global_state", {}).get("recent_hv_gain_mean", 0.0))
                ),
                "RecentHVGainStd": float(
                    safe_float(llm_state.get("global_state", {}).get("recent_hv_gain_std", 0.0))
                ),
                "AnyAcqFailureThisIter": int(any_acq_failure_this_iter),
                "NumAcqFailuresThisIter": int(num_acq_failures_this_iter),
                "SelectedProposalFailed": int(selected_proposal_failed),
                "SelectedProposalStatus": selected_proposal_status,
                "SelectedProposalError": selected_proposal_error,
                "TotalAcqFailures": int(total_acq_failures),
                "IterationsWithAnyAcqFailure": int(iterations_with_any_acq_failure),
                "SelectedProposalFailureCount": int(selected_proposal_failure_count),
                "RunHealthStatus": run_health_status,
            }
        )
        if bandit_policy is not None:
            rec["BanditReward"] = float(bandit_stats["reward"])
            rec["BanditRewardFeasibleTerm"] = float(bandit_stats["reward_feasible_term"])
            rec["BanditRewardHVTerm"] = float(bandit_stats["reward_hv_term"])
            rec["BanditHVScale"] = float(bandit_stats["hv_scale"])
            rec["BanditWarmStartFlag"] = int(bandit_policy.last_warmstart_flag)
            for arm in PORTFOLIO_ACQ_NAMES:
                rec[f"BanditUCB_{arm}"] = float(bandit_policy.last_ucb.get(arm, float("nan")))
                rec[f"BanditMean_{arm}"] = float(bandit_policy.means.get(arm, 0.0))
                rec[f"BanditCount_{arm}"] = int(bandit_policy.counts.get(arm, 0))
        else:
            rec["BanditReward"] = float("nan")
            rec["BanditRewardFeasibleTerm"] = float("nan")
            rec["BanditRewardHVTerm"] = float("nan")
            rec["BanditHVScale"] = float("nan")
            rec["BanditWarmStartFlag"] = 0
            for arm in PORTFOLIO_ACQ_NAMES:
                rec[f"BanditUCB_{arm}"] = float("nan")
                rec[f"BanditMean_{arm}"] = float("nan")
                rec[f"BanditCount_{arm}"] = 0

        for acq_name in ALL_ACQ_NAMES:
            diag = acq_diags_all.get(acq_name, {})
            rec[f"AcqValue_{acq_name}"] = diag.get("acq_value", float("nan"))
            rec[f"PredFeas_{acq_name}"] = diag.get("pred_feas_prob", float("nan"))
            rec[f"PredObjScore_{acq_name}"] = diag.get("pred_obj_score", float("nan"))
            rec[f"AcqStatus_{acq_name}"] = str(proposals.get(acq_name, {}).get("status", "unknown"))
            rec[f"AcqError_{acq_name}"] = str(proposals.get(acq_name, {}).get("error", ""))
            rec[f"Uncertainty_{acq_name}"] = diag.get("candidate_uncertainty", float("nan"))
            rec[f"Novelty_{acq_name}"] = diag.get("candidate_novelty", float("nan"))
            rec[f"HVProxy_{acq_name}"] = diag.get("posterior_mean_hv_proxy", float("nan"))
        for acq_name in PORTFOLIO_ACQ_NAMES:
            arm_ctx = llm_state.get("per_arm", {}).get(acq_name, {})
            rec[f"NormAcq_{acq_name}"] = float(
                safe_float(arm_ctx.get("normalized_acq_value", float("nan")), float("nan"))
            )
            rec[f"FeasRank_{acq_name}"] = int(arm_ctx.get("feasibility_rank", 0) or 0)
            rec[f"HVRank_{acq_name}"] = int(arm_ctx.get("hv_proxy_rank", 0) or 0)
            rec[f"UncertaintyRank_{acq_name}"] = int(arm_ctx.get("uncertainty_rank", 0) or 0)
        rows.append(rec)

        hv_prev = hv_now
        feasible_prev = feasible_now

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"campaign_{cfg.seed}.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    return int(cfg.seed)
