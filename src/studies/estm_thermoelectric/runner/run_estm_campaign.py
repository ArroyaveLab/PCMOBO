#!/usr/bin/env python3
"""Run one ESTM study campaign for a single policy across one or more seeds."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from core import IterationRecord, compact_json, ensure_dir, parse_seed_list, safe_float
from paths import ESTM_RUNS_ROOT
from policy import BanditUCBSwitchPolicy
from studies.estm_thermoelectric.constants import (
    ESTM_ARM_NAMES,
    ESTM_DATA_PATH,
    ESTM_FIXED_POLICY_NAMES,
    ESTM_MANIFEST_PATH,
    ESTM_POLICY_NAMES,
    POLICY_PROMPT_VERSION,
    is_estm_llm_policy,
    normalize_estm_policy_name,
)
from studies.estm_thermoelectric.policy.llm_switch import LLMSwitchPolicy
from studies.estm_thermoelectric.wrappers import (
    compute_hv_and_feasible_metrics,
    evaluate_selected_candidate,
    fit_predict_state,
    initial_seed_selection,
    load_problem,
    score_acquisition_candidates,
)


@dataclass
class ESTMSeedRunConfig:
    policy: str
    seeds: str = "1-5"
    iterations: int = 100
    n_init: int = 8
    workers: int = 1
    results_dir: str = ""
    data_path: str = str(ESTM_DATA_PATH)
    manifest_path: str = str(ESTM_MANIFEST_PATH)
    device: str = "auto"
    dtype: str = "float64"
    acq_mc_samples: int = 128
    acq_raw_samples: int = 256
    acq_num_restarts: int = 8
    acq_beta: float = 0.2
    acq_batch_eval_size: int = 1024
    llm_model: str = "gpt-4o"
    llm_provider: str = "openai"
    llm_timeout: int = 90
    memory_window: int = 10
    env_path: str = ".env"
    bandit_c: float = 1.0
    bandit_w_feas: float = 0.7
    bandit_w_hv: float = 0.3
    bandit_warmstart: int = 1

    def normalized_policy(self) -> str:
        return normalize_estm_policy_name(self.policy)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _acq_cfg(cfg: ESTMSeedRunConfig) -> Dict[str, Any]:
    return {
        "mc_samples": int(cfg.acq_mc_samples),
        "raw_samples": int(cfg.acq_raw_samples),
        "num_restarts": int(cfg.acq_num_restarts),
        "beta": float(cfg.acq_beta),
        "batch_eval_size": int(cfg.acq_batch_eval_size),
    }


def _normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {key: 0.5 for key in values}
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return {key: 0.5 for key in values}
    return {key: float((float(value) - lo) / (hi - lo)) for key, value in values.items()}


def cleanup_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _policy_acq_names(policy_name: str) -> tuple[str, ...]:
    normalized = normalize_estm_policy_name(policy_name)
    if normalized in ESTM_FIXED_POLICY_NAMES:
        return (normalized,)
    return tuple(ESTM_ARM_NAMES)


def _recent_arm_metrics(history: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    chosen = [item for item in history if str(item.get("chosen_acq", "")) == arm]
    iterations_since = (
        999999 if not chosen else int(history[-1]["iteration"] - chosen[-1]["iteration"])
    )
    if not chosen:
        return {
            "recent_count": 0,
            "recent_feasible_gain_rate": 0.0,
            "recent_hv_gain_mean": 0.0,
            "iterations_since_selected": iterations_since,
        }
    feasible_rate = sum(int(item.get("feasible_gain", 0) > 0) for item in chosen) / float(
        len(chosen)
    )
    hv_mean = sum(float(item.get("hv_scaled_gain", 0.0)) for item in chosen) / float(len(chosen))
    return {
        "recent_count": int(len(chosen)),
        "recent_feasible_gain_rate": float(feasible_rate),
        "recent_hv_gain_mean": float(hv_mean),
        "iterations_since_selected": int(iterations_since),
    }


def _same_acq_streak(history: List[Dict[str, Any]]) -> int:
    if not history:
        return 0
    last = str(history[-1].get("chosen_acq", ""))
    streak = 0
    for item in reversed(history):
        if str(item.get("chosen_acq", "")) != last:
            break
        streak += 1
    return streak


def _iterations_since_switch(history: List[Dict[str, Any]]) -> int:
    if len(history) <= 1:
        return 0
    last = str(history[-1].get("chosen_acq", ""))
    count = 0
    for item in reversed(history[:-1]):
        if str(item.get("chosen_acq", "")) != last:
            break
        count += 1
    return count


def _current_arm_no_hv_gain_streak(history: List[Dict[str, Any]]) -> int:
    if not history:
        return 0
    last = str(history[-1].get("chosen_acq", ""))
    count = 0
    for item in reversed(history):
        if str(item.get("chosen_acq", "")) != last:
            break
        if safe_float(item.get("hv_scaled_gain", 0.0)) > 0.0:
            break
        count += 1
    return count


def _best_arm(values: Dict[str, float]) -> tuple[str, float]:
    if not values:
        return "", 0.0
    arm = max(values.items(), key=lambda item: (safe_float(item[1], float("-inf")), item[0]))[0]
    return arm, safe_float(values.get(arm, 0.0))


def _rank_map(values: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(
        values.items(), key=lambda item: (safe_float(item[1], float("-inf")), item[0]), reverse=True
    )
    return {arm: idx + 1 for idx, (arm, _) in enumerate(ordered)}


def _build_llm_state(
    cfg: ESTMSeedRunConfig,
    *,
    iteration: int,
    proposals: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    observed_count: int,
    cumulative_feasible_count: int,
    current_hv_scaled: float,
    init_hv_scaled: float,
) -> Dict[str, Any]:
    acq_values = {arm: safe_float(p.get("acq_value", 0.0)) for arm, p in proposals.items()}
    acq_norm = _normalize_scores(acq_values)
    pred_feas = {arm: safe_float(p.get("pred_feas_prob", 0.0)) for arm, p in proposals.items()}
    hv_proxy = {
        arm: safe_float(p.get("posterior_mean_hv_proxy", 0.0)) for arm, p in proposals.items()
    }
    uncertainty = {
        arm: safe_float(p.get("candidate_uncertainty", 0.0)) for arm, p in proposals.items()
    }
    best_feas_arm, best_feas_value = _best_arm(pred_feas)
    best_hv_arm, best_hv_value = _best_arm(hv_proxy)
    best_uncertainty_arm, best_uncertainty_value = _best_arm(uncertainty)
    feas_rank = _rank_map(pred_feas)
    hv_rank = _rank_map(hv_proxy)
    uncertainty_rank = _rank_map(uncertainty)
    recent = history[-max(1, int(cfg.memory_window)) :]
    hv_gains = [safe_float(item.get("hv_scaled_gain", 0.0)) for item in recent]
    feas_flags = [int(item.get("feasible_gain", 0) > 0) for item in recent]
    recent_hv_mean = sum(hv_gains) / float(len(hv_gains)) if hv_gains else 0.0
    recent_hv_std = (
        float(pd.Series(hv_gains, dtype=float).std(ddof=0)) if len(hv_gains) > 1 else 0.0
    )
    recent_feas_rate = sum(feas_flags) / float(len(feas_flags)) if feas_flags else 0.0
    last_selected = str(history[-1].get("chosen_acq", "")) if history else ""
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm, proposal in proposals.items():
        arm_history = _recent_arm_metrics(recent, arm)
        per_arm[arm] = {
            "predicted_feasibility": safe_float(proposal.get("pred_feas_prob", 0.0)),
            "pred_obj_score": safe_float(proposal.get("pred_obj_score", 0.0)),
            "normalized_acq_value": safe_float(acq_norm.get(arm, 0.5), 0.5),
            "posterior_mean_hv_proxy": safe_float(proposal.get("posterior_mean_hv_proxy", 0.0)),
            "candidate_uncertainty": safe_float(proposal.get("candidate_uncertainty", 0.0)),
            "candidate_novelty": safe_float(proposal.get("candidate_novelty", 0.0)),
            **arm_history,
            "feasibility_rank": int(feas_rank.get(arm, len(proposals))),
            "hv_proxy_rank": int(hv_rank.get(arm, len(proposals))),
            "uncertainty_rank": int(uncertainty_rank.get(arm, len(proposals))),
        }
    return {
        "metadata": {
            "benchmark": "estm_thermoelectric",
            "iteration": int(iteration),
            "total_iterations": int(cfg.iterations),
            "allowed_arms": list(ESTM_ARM_NAMES),
        },
        "global_state": {
            "progress_ratio": float(iteration / max(1, int(cfg.iterations))),
            "cumulative_feasible_count": int(cumulative_feasible_count),
            "cumulative_observed_count": int(observed_count),
            "current_hv_scaled_fixed_range": float(current_hv_scaled),
            "hv_gain_since_init": float(current_hv_scaled - init_hv_scaled),
            "observed_feasible_fraction": float(cumulative_feasible_count / max(1, observed_count)),
            "recent_feasible_gain_rate": float(recent_feas_rate),
            "recent_hv_gain_mean": float(recent_hv_mean),
            "recent_hv_gain_std": float(recent_hv_std),
            "last_selected_acq": last_selected,
            "same_acq_streak": int(_same_acq_streak(history)),
            "current_arm_no_hv_gain_streak": int(_current_arm_no_hv_gain_streak(history)),
            "iterations_since_last_switch": int(_iterations_since_switch(history)),
        },
        "per_arm": per_arm,
        "recent_history": recent,
    }


def _decision_for_policy(
    cfg: ESTMSeedRunConfig,
    *,
    policy_name: str,
    iteration: int,
    llm_policy: LLMSwitchPolicy | None,
    bandit_policy: BanditUCBSwitchPolicy | None,
    llm_state: Dict[str, Any],
    proposals: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if policy_name in ESTM_FIXED_POLICY_NAMES:
        return {
            "acq": policy_name,
            "reason": f"fixed_policy_{policy_name}",
            "reflection": f"fixed policy selected {policy_name}",
            "confidence": 1.0,
            "source": "fixed_policy",
            "prompt_version": POLICY_PROMPT_VERSION,
            "llm_pipeline_mode": "three_agent",
            "llm_model_requests_this_decision": 0,
            "llm_stage_error": "",
            "llm_feasibility_advocate_json": "",
            "llm_hv_advocate_json": "",
            "llm_arbiter_json": "",
            "llm_payload_transmitted": 0,
            "llm_policy_name": "",
        }
    if policy_name == "bandit_ucb_switch":
        assert bandit_policy is not None
        decision = bandit_policy.select(iteration=iteration, acq_diagnostics=proposals)
        decision["prompt_version"] = POLICY_PROMPT_VERSION
        decision["llm_policy_name"] = ""
        decision["llm_payload_transmitted"] = 0
        return decision
    assert llm_policy is not None
    return llm_policy.decide(llm_state)


def _row_for_iteration(
    cfg: ESTMSeedRunConfig,
    *,
    seed: int,
    problem: Any,
    iteration: int,
    decision: Dict[str, Any],
    proposals: Dict[str, Dict[str, Any]],
    selected_proposal: Dict[str, Any],
    event: Dict[str, Any],
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    memory_snapshot_before_decision: Dict[str, Any],
    llm_payload_before_decision: Dict[str, Any] | None,
    bandit_policy: BanditUCBSwitchPolicy | None,
    bandit_update: Dict[str, float],
    cumulative_llm_requests: int,
    cumulative_llm_model_requests: int,
    total_acq_failures: int,
    iterations_with_any_acq_failure: int,
    observed_count: int,
) -> Dict[str, Any]:
    hv_gain = float(after_metrics["hypervolume_raw"] - before_metrics["hypervolume_raw"])
    hv_scaled_gain = float(
        after_metrics["hypervolume_scaled_fixed_range"]
        - before_metrics["hypervolume_scaled_fixed_range"]
    )
    feasible_gain = int(
        after_metrics["cumulative_feasible_count"] - before_metrics["cumulative_feasible_count"]
    )
    x_chosen = compact_json(event["x_selected"].detach().cpu().tolist(), max_chars=600)
    y_observed = compact_json(event["y_obj_observed"].detach().cpu().tolist(), max_chars=600)
    c_observed = compact_json(event["y_con_observed"].detach().cpu().tolist(), max_chars=600)
    generic = IterationRecord(
        iteration=iteration,
        selected_acq=str(decision["acq"]),
        selected_source=str(decision["source"]),
        feasible_observed=int(event["observed_feasible"]),
        cumulative_feasible_count=int(after_metrics["cumulative_feasible_count"]),
        hypervolume=float(after_metrics["hypervolume_raw"]),
        reason=str(decision["reason"]),
        reflection=str(decision["reflection"]),
        confidence=float(decision["confidence"]),
        memory_snapshot=compact_json(memory_snapshot_before_decision, max_chars=3000),
    ).as_dict()
    row = {
        **generic,
        "Benchmark": problem.name,
        "Policy": cfg.normalized_policy(),
        "Seed": int(seed),
        "Dim": int(problem.dim),
        "NumObjectives": int(problem.num_objectives),
        "NumConstraints": int(problem.num_constraints),
        "SelectedIndex": int(event["selected_index"]),
        "SelectedFormula": str(event.get("formula", "")),
        "SelectedTemperatureK": float(event.get("temperature_k", 0.0)),
        "SelectedReference": str(event.get("source_reference", "")),
        "XChosen": x_chosen,
        "YObserved": y_observed,
        "CObserved": c_observed,
        "HypervolumeRaw": float(after_metrics["hypervolume_raw"]),
        "HVGain": float(hv_gain),
        "HypervolumeScaledFixedRange": float(after_metrics["hypervolume_scaled_fixed_range"]),
        "HypervolumeScaledFixedRangeGain": float(hv_scaled_gain),
        "FeasibleGain": int(feasible_gain),
        "CumulativeObservedCount": int(observed_count),
        "CumulativeMeasured": int(observed_count),
        "CumulativePhaseObserved": int(observed_count),
        "FullMeasurementObserved": int(event["full_measurement_observed"]),
    }
    for arm in ESTM_ARM_NAMES:
        proposal = proposals.get(arm, {})
        row[f"AcqValue_{arm}"] = float(proposal.get("acq_value", 0.0) or 0.0)
        row[f"PredFeas_{arm}"] = float(proposal.get("pred_feas_prob", 0.0) or 0.0)
        row[f"PredObjScore_{arm}"] = float(proposal.get("pred_obj_score", 0.0) or 0.0)
        row[f"CandidateUncertainty_{arm}"] = float(
            proposal.get("candidate_uncertainty", 0.0) or 0.0
        )
        row[f"CandidateNovelty_{arm}"] = float(proposal.get("candidate_novelty", 0.0) or 0.0)
        row[f"PosteriorMeanHVProxy_{arm}"] = float(
            proposal.get("posterior_mean_hv_proxy", 0.0) or 0.0
        )
        row[f"AcqStatus_{arm}"] = str(proposal.get("status", "missing"))
        row[f"AcqError_{arm}"] = str(proposal.get("error", ""))
    bandit_fields = {
        "BanditReward": float(bandit_update.get("reward", 0.0)),
        "BanditRewardFeasibleTerm": float(bandit_update.get("reward_feasible_term", 0.0)),
        "BanditRewardHVTerm": float(bandit_update.get("reward_hv_term", 0.0)),
        "BanditHVScale": float(bandit_update.get("hv_scale", 1.0)),
        "BanditWarmStartFlag": int(
            bandit_policy.last_warmstart_flag if bandit_policy is not None else 0
        ),
    }
    for arm in ESTM_ARM_NAMES:
        bandit_fields[f"BanditUCB_{arm}"] = safe_float(
            (bandit_policy.last_ucb if bandit_policy else {}).get(arm, float("nan")),
            float("nan"),
        )
        bandit_fields[f"BanditMean_{arm}"] = safe_float(
            (bandit_policy.means if bandit_policy else {}).get(arm, 0.0)
        )
        bandit_fields[f"BanditCount_{arm}"] = int(
            (bandit_policy.counts if bandit_policy else {}).get(arm, 0)
        )
    row.update(
        {
            **bandit_fields,
            "DecisionPromptVersion": str(decision.get("prompt_version", POLICY_PROMPT_VERSION)),
            "LLMPolicyName": str(decision.get("llm_policy_name", "")),
            "LLMPayloadBeforeDecision": (
                json.dumps(llm_payload_before_decision, separators=(",", ":"), ensure_ascii=True)
                if llm_payload_before_decision is not None
                else ""
            ),
            "LLMPayloadTransmitted": int(decision.get("llm_payload_transmitted", 0) or 0),
            "LLMDecisionCount": int(cumulative_llm_requests),
            "LLMModelRequestCount": int(cumulative_llm_model_requests),
            "LLMPipelineMode": str(decision.get("llm_pipeline_mode", "three_agent")),
            "LLMModelRequestsThisDecision": int(
                decision.get("llm_model_requests_this_decision", 0) or 0
            ),
            "LLMFeasibilityAdvocateJSON": str(decision.get("llm_feasibility_advocate_json", "")),
            "LLMHVAdvocateJSON": str(decision.get("llm_hv_advocate_json", "")),
            "LLMArbiterJSON": str(decision.get("llm_arbiter_json", "")),
            "LLMStageError": str(decision.get("llm_stage_error", "")),
            "SelectedProposalFailed": int(selected_proposal.get("status", "ok") != "ok"),
            "TotalAcqFailures": int(total_acq_failures),
            "IterationsWithAnyAcqFailure": int(iterations_with_any_acq_failure),
            "RunHealthStatus": "degraded" if total_acq_failures > 0 else "valid",
        }
    )
    return row


def _run_seed(seed: int, cfg: ESTMSeedRunConfig, llm_policy_factory: Any = None) -> str:
    _seed_everything(seed)
    problem = load_problem(cfg, device=cfg.device, dtype=cfg.dtype)
    results_dir = ensure_dir(cfg.results_dir)
    out_file = results_dir / f"campaign_{int(seed)}.csv"
    init_indices = [int(idx) for idx in initial_seed_selection(problem, seed, cfg.n_init).tolist()]
    observed_indices = set(init_indices)
    selected_indices = set(init_indices)
    history: List[Dict[str, Any]] = []
    bandit_policy = None
    llm_policy = None
    if cfg.normalized_policy() == "bandit_ucb_switch":
        bandit_policy = BanditUCBSwitchPolicy(
            arms=ESTM_ARM_NAMES,
            exploration_c=float(cfg.bandit_c),
            weight_feasible=float(cfg.bandit_w_feas),
            weight_hv=float(cfg.bandit_w_hv),
            warmstart=bool(int(cfg.bandit_warmstart)),
        )
    if is_estm_llm_policy(cfg.normalized_policy()):
        factory = llm_policy_factory or LLMSwitchPolicy
        llm_policy = factory(
            model=cfg.llm_model,
            llm_provider=cfg.llm_provider,
            env_path=cfg.env_path,
            timeout_s=int(cfg.llm_timeout),
            memory_window=int(cfg.memory_window),
        )
    campaign_rows: List[Dict[str, Any]] = []
    init_metrics = compute_hv_and_feasible_metrics(problem, observed_indices)
    cumulative_llm_requests = 0
    cumulative_llm_model_requests = 0
    total_acq_failures = 0
    iterations_with_any_acq_failure = 0
    acq_names = _policy_acq_names(cfg.normalized_policy())
    for iteration in range(1, int(cfg.iterations) + 1):
        remaining = [
            idx for idx in range(int(problem.x_all.shape[0])) if idx not in selected_indices
        ]
        if not remaining:
            break
        before_metrics = compute_hv_and_feasible_metrics(problem, observed_indices)
        model_state = fit_predict_state(problem, observed_indices)
        proposals = score_acquisition_candidates(
            problem, model_state, remaining, acq_names, _acq_cfg(cfg)
        )
        llm_state_before = _build_llm_state(
            cfg,
            iteration=iteration,
            proposals=proposals,
            history=history,
            observed_count=len(observed_indices),
            cumulative_feasible_count=int(before_metrics["cumulative_feasible_count"]),
            current_hv_scaled=float(before_metrics["hypervolume_scaled_fixed_range"]),
            init_hv_scaled=float(init_metrics["hypervolume_scaled_fixed_range"]),
        )
        llm_payload_before = (
            llm_policy.build_audit_payload(llm_state_before) if llm_policy is not None else None
        )
        decision = _decision_for_policy(
            cfg,
            policy_name=cfg.normalized_policy(),
            iteration=iteration,
            llm_policy=llm_policy,
            bandit_policy=bandit_policy,
            llm_state=llm_state_before,
            proposals=proposals,
        )
        chosen_arm = str(decision["acq"]).strip().lower()
        selected_proposal = proposals[chosen_arm]
        event = evaluate_selected_candidate(problem, int(selected_proposal["candidate_index"]))
        selected_indices.add(int(event["selected_index"]))
        observed_indices.add(int(event["selected_index"]))
        after_metrics = compute_hv_and_feasible_metrics(problem, observed_indices)
        hv_scaled_gain = float(
            after_metrics["hypervolume_scaled_fixed_range"]
            - before_metrics["hypervolume_scaled_fixed_range"]
        )
        feasible_gain = int(
            after_metrics["cumulative_feasible_count"] - before_metrics["cumulative_feasible_count"]
        )
        bandit_update = {
            "reward": 0.0,
            "reward_feasible_term": 0.0,
            "reward_hv_term": 0.0,
            "hv_scale": 1.0,
        }
        if bandit_policy is not None:
            bandit_update = bandit_policy.update(chosen_arm, feasible_gain, hv_scaled_gain)
        any_failure = any(
            str(proposal.get("status", "ok")) != "ok" for proposal in proposals.values()
        )
        total_acq_failures += sum(
            int(str(proposal.get("status", "ok")) != "ok") for proposal in proposals.values()
        )
        iterations_with_any_acq_failure += int(any_failure)
        if str(decision.get("source", "")) == "llm_three_agent":
            cumulative_llm_requests += 1
        cumulative_llm_model_requests += int(
            decision.get("llm_model_requests_this_decision", 0) or 0
        )

        chosen_proposal = proposals[chosen_arm]
        best_feas_arm, best_feas_value = _best_arm(
            {arm: p.get("pred_feas_prob", 0.0) for arm, p in proposals.items()}
        )
        best_hv_arm, best_hv_value = _best_arm(
            {arm: p.get("posterior_mean_hv_proxy", 0.0) for arm, p in proposals.items()}
        )
        best_uncertainty_arm, best_uncertainty_value = _best_arm(
            {arm: p.get("candidate_uncertainty", 0.0) for arm, p in proposals.items()}
        )
        chosen_feas = float(chosen_proposal.get("pred_feas_prob", 0.0))
        chosen_hv_proxy = float(chosen_proposal.get("posterior_mean_hv_proxy", 0.0))
        history.append(
            {
                "iteration": int(iteration),
                "chosen_acq": chosen_arm,
                "observed_feasible": int(event["observed_feasible"]),
                "feasible_gain": int(feasible_gain),
                "hv_scaled_gain": float(hv_scaled_gain),
                "cumulative_feasible_count": int(after_metrics["cumulative_feasible_count"]),
                "chosen_predicted_feasibility": chosen_feas,
                "chosen_hv_proxy": float(chosen_hv_proxy),
                "chosen_uncertainty": float(chosen_proposal.get("candidate_uncertainty", 0.0)),
                "chosen_novelty": float(chosen_proposal.get("candidate_novelty", 0.0)),
                "best_feas_arm": best_feas_arm,
                "best_feas_value": float(best_feas_value),
                "best_hv_arm": best_hv_arm,
                "best_hv_value": float(best_hv_value),
                "best_uncertainty_arm": best_uncertainty_arm,
                "best_uncertainty_value": float(best_uncertainty_value),
                "chosen_vs_best_feas_gap": float(best_feas_value - chosen_feas),
                "chosen_vs_best_hv_gap": float(best_hv_value - chosen_hv_proxy),
            }
        )
        row = _row_for_iteration(
            cfg,
            seed=seed,
            problem=problem,
            iteration=iteration,
            decision=decision,
            proposals=proposals,
            selected_proposal=selected_proposal,
            event=event,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            memory_snapshot_before_decision=llm_state_before,
            llm_payload_before_decision=llm_payload_before,
            bandit_policy=bandit_policy,
            bandit_update=bandit_update,
            cumulative_llm_requests=cumulative_llm_requests,
            cumulative_llm_model_requests=cumulative_llm_model_requests,
            total_acq_failures=total_acq_failures,
            iterations_with_any_acq_failure=iterations_with_any_acq_failure,
            observed_count=len(observed_indices),
        )
        campaign_rows.append(row)
        pd.DataFrame(campaign_rows).to_csv(out_file, index=False)
        del (
            model_state,
            proposals,
            llm_state_before,
            llm_payload_before,
            decision,
            selected_proposal,
            event,
        )
        del before_metrics, after_metrics, row, chosen_proposal, bandit_update
        cleanup_runtime_memory()
    pd.DataFrame(campaign_rows).to_csv(out_file, index=False)
    return str(out_file)


def run_campaign(cfg: ESTMSeedRunConfig, llm_policy_factory: Any = None) -> Path:
    policy = cfg.normalized_policy()
    if policy not in ESTM_POLICY_NAMES:
        raise ValueError(f"Unsupported ESTM policy: {cfg.policy}")
    results_dir = Path(cfg.results_dir) if cfg.results_dir else ESTM_RUNS_ROOT / "default" / policy
    cfg.results_dir = str(ensure_dir(results_dir))
    seeds = parse_seed_list(cfg.seeds)
    run_config = asdict(cfg)
    run_config["normalized_policy"] = str(policy)
    run_config["is_llm_policy"] = bool(is_estm_llm_policy(policy))
    (results_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if int(cfg.workers) <= 1 or llm_policy_factory is not None:
        for seed in seeds:
            _run_seed(seed, cfg, llm_policy_factory=llm_policy_factory)
            cleanup_runtime_memory()
    else:
        with ProcessPoolExecutor(max_workers=int(cfg.workers)) as pool:
            futures = [pool.submit(_run_seed, seed, cfg, None) for seed in seeds]
            for future in as_completed(futures):
                future.result()
        cleanup_runtime_memory()
    cleanup_runtime_memory()
    return results_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seeds", default="1-5")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--data-path", default=str(ESTM_DATA_PATH))
    parser.add_argument("--manifest-path", default=str(ESTM_MANIFEST_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--acq-mc-samples", type=int, default=128)
    parser.add_argument("--acq-raw-samples", type=int, default=256)
    parser.add_argument("--acq-num-restarts", type=int, default=8)
    parser.add_argument("--acq-beta", type=float, default=0.2)
    parser.add_argument("--acq-batch-eval-size", type=int, default=1024)
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-timeout", type=int, default=90)
    parser.add_argument("--memory-window", type=int, default=10)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--bandit-c", type=float, default=1.0)
    parser.add_argument("--bandit-w-feas", type=float, default=0.7)
    parser.add_argument("--bandit-w-hv", type=float, default=0.3)
    parser.add_argument("--bandit-warmstart", type=int, default=1)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    cfg = ESTMSeedRunConfig(
        policy=ns.policy,
        seeds=ns.seeds,
        iterations=int(ns.iterations),
        n_init=int(ns.n_init),
        workers=int(ns.workers),
        results_dir=str(ns.results_dir),
        data_path=str(ns.data_path),
        manifest_path=str(ns.manifest_path),
        device=ns.device,
        dtype=ns.dtype,
        acq_mc_samples=int(ns.acq_mc_samples),
        acq_raw_samples=int(ns.acq_raw_samples),
        acq_num_restarts=int(ns.acq_num_restarts),
        acq_beta=float(ns.acq_beta),
        acq_batch_eval_size=int(ns.acq_batch_eval_size),
        llm_model=ns.llm_model,
        llm_provider=ns.llm_provider,
        llm_timeout=int(ns.llm_timeout),
        memory_window=int(ns.memory_window),
        env_path=ns.env_path,
        bandit_c=float(ns.bandit_c),
        bandit_w_feas=float(ns.bandit_w_feas),
        bandit_w_hv=float(ns.bandit_w_hv),
        bandit_warmstart=int(ns.bandit_warmstart),
    )
    run_campaign(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
