"""Model fitting and finite-pool acquisition scoring for the HEA study."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch.distributions import Normal

from acq import AcqState, build_acquisition
from engine.modeling import (
    fit_model_bundle,
    fit_output_model,
    weighted_obj_score,
)
from studies.hea_design.constants import HEA_ARM_NAMES
from studies.hea_design.wrappers.hea_problem import (
    HEAProblem,
    compute_hv_and_feasible_metrics,
    evaluate_selected_candidate,
    initial_seed_selection,
    load_problem,
)


def _matrix(values: Any) -> Any:
    if int(values.ndim) == 0:
        return values.reshape(1, 1)
    if int(values.ndim) == 1:
        return values.reshape(1, int(values.shape[-1]))
    return values.reshape(-1, int(values.shape[-1]))


def fit_predict_state(
    problem: HEAProblem, measured_indices: Iterable[int], phase_observed_indices: Iterable[int]
) -> Dict[str, Any]:
    measured = [int(idx) for idx in measured_indices]
    phase = [int(idx) for idx in phase_observed_indices]
    if not measured:
        raise ValueError("HEA study requires at least one measured point.")
    if not phase:
        raise ValueError("HEA study requires at least one phase-observed point.")
    measured_t = torch.tensor(measured, device=problem.x_all.device, dtype=torch.long)
    phase_t = torch.tensor(phase, device=problem.x_all.device, dtype=torch.long)
    x_measured = problem.x_all[measured_t]
    y_obj = problem.y_obj_all[measured_t]
    y_con_obj = problem.y_con_obj_all[measured_t]
    x_phase = problem.x_all[phase_t]
    y_phase = problem.bcc_single_mask[phase_t].to(dtype=problem.y_obj_all.dtype).unsqueeze(-1)
    bundle = fit_model_bundle(x_measured, y_obj, y_con_obj)
    model_phase = fit_output_model(x_phase, y_phase)
    return {
        "x_obs": x_measured,
        "y_obj": y_obj,
        "y_con_obj": y_con_obj,
        "bundle": bundle,
        "model_phase": model_phase,
    }


def _pass_probabilities(
    problem: HEAProblem, mu_obj: torch.Tensor, sigma_obj: torch.Tensor
) -> Dict[str, torch.Tensor]:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=mu_obj.device, dtype=mu_obj.dtype),
        torch.tensor(1.0, device=mu_obj.device, dtype=mu_obj.dtype),
    )
    return {
        "p_st": (
            1.0
            - normal.cdf(
                (float(problem.thresholds["st"]) - mu_obj[:, 0]) / sigma_obj[:, 0].clamp_min(1e-9)
            )
        ).clamp(1e-9, 1.0),
        "p_den": normal.cdf(
            (float(problem.thresholds["density"]) - (-mu_obj[:, 1]))
            / sigma_obj[:, 1].clamp_min(1e-9)
        ).clamp(1e-9, 1.0),
        "p_ys": (
            1.0
            - normal.cdf(
                (float(problem.thresholds["ys"]) - mu_obj[:, 2]) / sigma_obj[:, 2].clamp_min(1e-9)
            )
        ).clamp(1e-9, 1.0),
        "p_pugh": (
            1.0
            - normal.cdf(
                (float(problem.thresholds["pugh"]) - mu_obj[:, 3]) / sigma_obj[:, 3].clamp_min(1e-9)
            )
        ).clamp(1e-9, 1.0),
    }


def _phase_probability(
    model_phase: Any, x_pool: torch.Tensor, batch_size: int | None = None
) -> torch.Tensor:
    chunk = int(batch_size or int(x_pool.shape[0]) or 1)
    parts = []
    with torch.no_grad():
        for start in range(0, int(x_pool.shape[0]), chunk):
            x_batch = x_pool[start : start + chunk]
            phase_post = model_phase.posterior(x_batch)
            parts.append(_matrix(phase_post.mean).reshape(-1))
    if not parts:
        return torch.empty(0, device=x_pool.device, dtype=x_pool.dtype)
    return torch.sigmoid(torch.cat(parts, dim=0)).clamp(1e-9, 1.0)


def _pool_diagnostics(
    problem: HEAProblem,
    model_state: Dict[str, Any],
    x_pool: torch.Tensor,
    batch_size: int | None = None,
) -> Dict[str, torch.Tensor]:
    chunk = int(batch_size or int(x_pool.shape[0]) or 1)
    mu_parts = []
    sigma_parts = []
    obj_feas_parts = []
    bcc_parts = []
    total_feas_parts = []
    obj_score_parts = []
    with torch.no_grad():
        for start in range(0, int(x_pool.shape[0]), chunk):
            x_batch = x_pool[start : start + chunk]
            obj_post = model_state["bundle"].model_obj.posterior(x_batch)
            mu_obj = _matrix(obj_post.mean)
            sigma_obj = _matrix(obj_post.variance).clamp_min(1e-12).sqrt()
            probs = _pass_probabilities(problem, mu_obj, sigma_obj)
            pred_obj_feas_prob = probs["p_st"] * probs["p_den"] * probs["p_ys"] * probs["p_pugh"]
            pred_bcc_prob = _phase_probability(
                model_state["model_phase"], x_batch, batch_size=chunk
            )
            pred_total_feas_prob = pred_obj_feas_prob * pred_bcc_prob
            pred_obj_score = weighted_obj_score(mu_obj)
            mu_parts.append(mu_obj)
            sigma_parts.append(sigma_obj)
            obj_feas_parts.append(pred_obj_feas_prob)
            bcc_parts.append(pred_bcc_prob)
            total_feas_parts.append(pred_total_feas_prob)
            obj_score_parts.append(pred_obj_score)
    if not mu_parts:
        empty_matrix = torch.empty(
            (0, int(problem.num_objectives)), device=x_pool.device, dtype=x_pool.dtype
        )
        empty_vector = torch.empty(0, device=x_pool.device, dtype=x_pool.dtype)
        return {
            "mu_obj": empty_matrix,
            "sigma_obj": empty_matrix.clone(),
            "pred_obj_feas_prob": empty_vector,
            "pred_bcc_prob": empty_vector,
            "pred_total_feas_prob": empty_vector,
            "pred_obj_score": empty_vector,
        }
    return {
        "mu_obj": torch.cat(mu_parts, dim=0),
        "sigma_obj": torch.cat(sigma_parts, dim=0),
        "pred_obj_feas_prob": torch.cat(obj_feas_parts, dim=0),
        "pred_bcc_prob": torch.cat(bcc_parts, dim=0),
        "pred_total_feas_prob": torch.cat(total_feas_parts, dim=0),
        "pred_obj_score": torch.cat(obj_score_parts, dim=0),
    }


def _score_pool_acqf(acqf: Any, x_pool: torch.Tensor, batch_size: int) -> torch.Tensor:
    values = []
    chunk = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, int(x_pool.shape[0]), chunk):
            x_batch = x_pool[start : start + chunk].unsqueeze(-2)
            out = acqf(x_batch)
            values.append(out.reshape(-1))
    return (
        torch.cat(values, dim=0)
        if values
        else torch.empty(0, device=x_pool.device, dtype=x_pool.dtype)
    )


def _full_design_objective_means(
    problem: HEAProblem,
    model_state: Dict[str, Any],
    batch_size: int | None = None,
) -> torch.Tensor:
    chunk = int(batch_size or int(problem.x_all.shape[0]) or 1)
    parts = []
    with torch.no_grad():
        for start in range(0, int(problem.x_all.shape[0]), chunk):
            x_batch = problem.x_all[start : start + chunk]
            obj_post = model_state["bundle"].model_obj.posterior(x_batch)
            parts.append(_matrix(obj_post.mean))
    if not parts:
        return torch.empty(
            (0, int(problem.num_objectives)), device=problem.x_all.device, dtype=problem.x_all.dtype
        )
    return torch.cat(parts, dim=0)


def _pehvi_reference_point(problem: HEAProblem) -> torch.Tensor:
    # Custom HEA pEHVI uses a fixed maximize-space reference [ST, -Density, YS, Pugh].
    return torch.tensor(
        [0.0, -30.0, 0.0, 0.0], device=problem.x_all.device, dtype=problem.x_all.dtype
    ).reshape(1, -1)


def _shifted_spread(values: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if int(values.shape[0]) == 0:
        return torch.zeros(int(ref.shape[-1]), device=ref.device, dtype=ref.dtype)
    shifted = values - ref
    return shifted.max(dim=0).values - shifted.min(dim=0).values


def _observed_feasible_front(model_state: Dict[str, Any]) -> torch.Tensor:
    y_obj = model_state["y_obj"]
    y_con = model_state["y_con_obj"]
    feasible = (y_con <= 0.0).all(dim=-1)
    if not bool(feasible.any()):
        return y_obj.new_empty((0, int(y_obj.shape[-1])))
    feasible_y = y_obj[feasible]
    keep = torch.ones(int(feasible_y.shape[0]), dtype=torch.bool, device=feasible_y.device)
    for i in range(int(feasible_y.shape[0])):
        if not bool(keep[i]):
            continue
        dominates = (feasible_y >= feasible_y[i]).all(dim=-1) & (feasible_y > feasible_y[i]).any(
            dim=-1
        )
        dominates[i] = False
        if bool(dominates.any()):
            keep[i] = False
    return feasible_y[keep]


def _pehvi_ranges(
    problem: HEAProblem,
    model_state: Dict[str, Any],
    diagnostics: Dict[str, torch.Tensor],
    batch_size: int | None = None,
) -> torch.Tensor:
    # Dynamic scaling uses full-design posterior mean spread with numerical guards.
    ref = _pehvi_reference_point(problem)
    full_means = _full_design_objective_means(problem, model_state, batch_size=batch_size)
    full_spread = _shifted_spread(full_means, ref)
    pool_spread = _shifted_spread(diagnostics["mu_obj"], ref)
    ones = torch.ones_like(pool_spread if int(pool_spread.numel()) else full_spread)

    ranges = full_spread
    invalid_full = (~torch.isfinite(ranges)) | (ranges <= 1e-12)
    if bool(invalid_full.any()):
        ranges = torch.where(invalid_full, pool_spread, ranges)

    invalid_pool = (~torch.isfinite(ranges)) | (ranges <= 1e-12)
    if bool(invalid_pool.any()):
        ranges = torch.where(invalid_pool, ones, ranges)
    return ranges.clamp_min(1e-12)


def _pehvi_scores(
    problem: HEAProblem,
    model_state: Dict[str, Any],
    diagnostics: Dict[str, torch.Tensor],
    ranges_dyn: torch.Tensor,
) -> torch.Tensor:
    mu_obj = diagnostics["mu_obj"]
    sigma_obj = diagnostics["sigma_obj"].clamp_min(1e-12)
    if int(mu_obj.shape[0]) == 0:
        return torch.empty(0, device=mu_obj.device, dtype=mu_obj.dtype)

    ref = _pehvi_reference_point(problem)
    means_scaled = (mu_obj - ref) / ranges_dyn
    sigmas_scaled = sigma_obj / ranges_dyn

    normal = Normal(
        torch.tensor(0.0, device=mu_obj.device, dtype=mu_obj.dtype),
        torch.tensor(1.0, device=mu_obj.device, dtype=mu_obj.dtype),
    )
    s_up = means_scaled / sigmas_scaled
    up = means_scaled * normal.cdf(s_up) + sigmas_scaled * torch.exp(normal.log_prob(s_up))
    box = torch.prod(up, dim=-1)

    pareto = _observed_feasible_front(model_state)
    if int(pareto.shape[0]) == 0:
        return box.clamp_min(0.0)

    shifted_pareto = (pareto - ref) / ranges_dyn
    pehvi = torch.full_like(box, float("inf"))
    for idx in range(int(shifted_pareto.shape[0])):
        p = shifted_pareto[idx : idx + 1]
        s_low = (means_scaled - p) / sigmas_scaled
        low = (means_scaled - p) * normal.cdf(s_low) + sigmas_scaled * torch.exp(
            normal.log_prob(s_low)
        )
        diff = (up - low).clamp_min(0.0)
        dominated_single = torch.prod(diff, dim=-1)
        ehvi_k = box - dominated_single
        pehvi = torch.minimum(pehvi, ehvi_k)
    return pehvi.clamp_min(0.0)


def _bcc_weighted_scores(base_scores: torch.Tensor, pred_bcc_prob: torch.Tensor) -> torch.Tensor:
    return base_scores * pred_bcc_prob


def score_acquisition_candidates(
    problem: HEAProblem,
    model_state: Dict[str, Any],
    candidate_indices: Iterable[int],
    acq_names: Iterable[str],
    acq_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    candidates = [int(idx) for idx in candidate_indices]
    if not candidates:
        raise ValueError("No remaining HEA candidates to score.")
    x_pool = problem.x_all[torch.tensor(candidates, device=problem.x_all.device, dtype=torch.long)]
    diagnostics = _pool_diagnostics(
        problem,
        model_state,
        x_pool,
        batch_size=int(acq_cfg.get("diagnostic_batch_size", acq_cfg.get("batch_eval_size", 1024))),
    )
    proposals: Dict[str, Dict[str, Any]] = {}
    pehvi_ranges: torch.Tensor | None = None
    for raw_name in acq_names:
        acq = str(raw_name).strip().lower()
        if acq == "pof":
            base_scores = diagnostics["pred_obj_feas_prob"]
        elif acq == "pehvi":
            if pehvi_ranges is None:
                pehvi_ranges = _pehvi_ranges(
                    problem,
                    model_state,
                    diagnostics,
                    batch_size=int(
                        acq_cfg.get("diagnostic_batch_size", acq_cfg.get("batch_eval_size", 1024))
                    ),
                )
            base_scores = _pehvi_scores(problem, model_state, diagnostics, pehvi_ranges)
        elif acq in {"qlognparego", "qucb"}:
            state = AcqState(
                x_obs=model_state["x_obs"],
                y_obj=model_state["y_obj"],
                y_con=model_state["y_con_obj"],
                bounds=problem.bounds,
                reference_point=problem.reference_point,
                q=1,
                mc_samples=int(acq_cfg.get("mc_samples", 128)),
                raw_samples=int(acq_cfg.get("raw_samples", 256)),
                num_restarts=int(acq_cfg.get("num_restarts", 8)),
                beta=float(acq_cfg.get("beta", 0.2)),
            )
            acqf = build_acquisition(acq, model_state["bundle"], state)
            base_scores = _score_pool_acqf(acqf, x_pool, int(acq_cfg.get("batch_eval_size", 512)))
        else:
            raise ValueError(f"Unknown HEA acquisition arm: {acq}")
        scores = _bcc_weighted_scores(base_scores, diagnostics["pred_bcc_prob"])
        best_pos = int(torch.argmax(scores).item())
        best_idx = int(candidates[best_pos])
        proposals[acq] = {
            "acq": acq,
            "candidate_index": best_idx,
            "candidate": problem.x_all[best_idx].detach().clone(),
            "base_acq_value": float(base_scores[best_pos].item()),
            "acq_value": float(scores[best_pos].item()),
            "pred_feas_prob": float(diagnostics["pred_total_feas_prob"][best_pos].item()),
            "pred_obj_feas_prob": float(diagnostics["pred_obj_feas_prob"][best_pos].item()),
            "pred_obj_score": float(diagnostics["pred_obj_score"][best_pos].item()),
            "pred_bcc_prob": float(diagnostics["pred_bcc_prob"][best_pos].item()),
            "pred_total_feas_prob": float(diagnostics["pred_total_feas_prob"][best_pos].item()),
            "status": "ok",
            "error": "",
        }
    return proposals


__all__ = [
    "HEAProblem",
    "HEA_ARM_NAMES",
    "load_problem",
    "initial_seed_selection",
    "fit_predict_state",
    "score_acquisition_candidates",
    "evaluate_selected_candidate",
    "compute_hv_and_feasible_metrics",
]
