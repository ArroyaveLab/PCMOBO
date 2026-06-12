"""Model fitting and finite-pool acquisition scoring for the ESTM study."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from acq import AcqState, build_acquisition
from engine.modeling import (
    compute_hv,
    fit_model_bundle,
    predict_constraint_feasibility,
    weighted_obj_score,
)
from studies.estm_thermoelectric.wrappers.estm_problem import ESTMProblem


def _matrix(values: Any) -> Any:
    if int(values.ndim) == 0:
        return values.reshape(1, 1)
    if int(values.ndim) == 1:
        return values.reshape(1, int(values.shape[-1]))
    return values.reshape(-1, int(values.shape[-1]))


def _objective_uncertainty_score(variance: torch.Tensor) -> torch.Tensor:
    std = _matrix(variance).clamp_min(1e-12).sqrt()
    return std.mean(dim=-1)


def _normalized_nn_distance(
    x_pool: torch.Tensor, x_obs: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    if int(x_obs.shape[0]) == 0:
        return torch.zeros(int(x_pool.shape[0]), device=x_pool.device, dtype=x_pool.dtype)
    span = (bounds[1] - bounds[0]).clamp_min(1e-12)
    pool_norm = (x_pool - bounds[0]) / span
    obs_norm = (x_obs - bounds[0]) / span
    dists = torch.cdist(pool_norm, obs_norm)
    return dists.min(dim=-1).values


def _posterior_mean_hv_proxy(
    y_obj: torch.Tensor,
    y_con: torch.Tensor,
    reference_point: torch.Tensor,
    candidate_mu_obj: torch.Tensor,
    pred_feas_prob: torch.Tensor,
) -> torch.Tensor:
    current_hv = compute_hv(y_obj, y_con, reference_point)
    q = int(candidate_mu_obj.shape[0])
    pseudo_con = -torch.ones(
        q,
        int(y_con.shape[-1]),
        device=y_con.device,
        dtype=y_con.dtype,
    )
    hv_aug = compute_hv(
        torch.cat([y_obj, candidate_mu_obj], dim=0),
        torch.cat([y_con, pseudo_con], dim=0),
        reference_point,
    )
    hv_delta = max(0.0, float(hv_aug - current_hv))
    return pred_feas_prob * hv_delta


def fit_predict_state(problem: ESTMProblem, observed_indices: Iterable[int]) -> Dict[str, Any]:
    observed = [int(idx) for idx in observed_indices]
    if not observed:
        raise ValueError("ESTM study requires at least one observed point.")
    observed_t = torch.tensor(observed, device=problem.x_all.device, dtype=torch.long)
    x_obs = problem.x_all[observed_t]
    y_obj = problem.y_obj_all[observed_t]
    y_con = problem.y_con_all[observed_t]
    bundle = fit_model_bundle(x_obs, y_obj, y_con)
    return {
        "x_obs": x_obs,
        "y_obj": y_obj,
        "y_con": y_con,
        "bundle": bundle,
    }


def _pool_diagnostics(
    problem: ESTMProblem, model_state: Dict[str, Any], x_pool: torch.Tensor
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        obj_post = model_state["bundle"].model_obj.posterior(x_pool)
        mu_obj = _matrix(obj_post.mean)
        pred_obj_score = weighted_obj_score(mu_obj)
        pred_feas_prob, _, _ = predict_constraint_feasibility(
            model_state["bundle"].model_con, x_pool
        )
        candidate_uncertainty = _objective_uncertainty_score(obj_post.variance)
        candidate_novelty = _normalized_nn_distance(x_pool, model_state["x_obs"], problem.bounds)
        hv_proxy = _posterior_mean_hv_proxy(
            y_obj=model_state["y_obj"],
            y_con=model_state["y_con"],
            reference_point=problem.reference_point,
            candidate_mu_obj=mu_obj,
            pred_feas_prob=pred_feas_prob,
        )
    return {
        "mu_obj": mu_obj,
        "pred_obj_score": pred_obj_score,
        "pred_feas_prob": pred_feas_prob,
        "candidate_uncertainty": candidate_uncertainty,
        "candidate_novelty": candidate_novelty,
        "posterior_mean_hv_proxy": hv_proxy,
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


def score_acquisition_candidates(
    problem: ESTMProblem,
    model_state: Dict[str, Any],
    candidate_indices: Iterable[int],
    acq_names: Iterable[str],
    acq_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    candidates = [int(idx) for idx in candidate_indices]
    if not candidates:
        raise ValueError("No remaining ESTM candidates to score.")
    x_pool = problem.x_all[torch.tensor(candidates, device=problem.x_all.device, dtype=torch.long)]
    diagnostics = _pool_diagnostics(problem, model_state, x_pool)
    proposals: Dict[str, Dict[str, Any]] = {}
    for raw_name in acq_names:
        acq = str(raw_name).strip().lower()
        state = AcqState(
            x_obs=model_state["x_obs"],
            y_obj=model_state["y_obj"],
            y_con=model_state["y_con"],
            bounds=problem.bounds,
            reference_point=problem.reference_point,
            q=1,
            mc_samples=int(acq_cfg.get("mc_samples", 128)),
            raw_samples=int(acq_cfg.get("raw_samples", 256)),
            num_restarts=int(acq_cfg.get("num_restarts", 8)),
            beta=float(acq_cfg.get("beta", 0.2)),
        )
        acqf = build_acquisition(acq, model_state["bundle"], state)
        scores = _score_pool_acqf(acqf, x_pool, int(acq_cfg.get("batch_eval_size", 512)))
        best_pos = int(torch.argmax(scores).item())
        best_idx = int(candidates[best_pos])
        proposals[acq] = {
            "acq": acq,
            "candidate_index": best_idx,
            "candidate": problem.x_all[best_idx].detach().clone(),
            "acq_value": float(scores[best_pos].item()),
            "pred_feas_prob": float(diagnostics["pred_feas_prob"][best_pos].item()),
            "pred_obj_score": float(diagnostics["pred_obj_score"][best_pos].item()),
            "candidate_uncertainty": float(diagnostics["candidate_uncertainty"][best_pos].item()),
            "candidate_novelty": float(diagnostics["candidate_novelty"][best_pos].item()),
            "posterior_mean_hv_proxy": float(
                diagnostics["posterior_mean_hv_proxy"][best_pos].item()
            ),
            "status": "ok",
            "error": "",
        }
    return proposals
