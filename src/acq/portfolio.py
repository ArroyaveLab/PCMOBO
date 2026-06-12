"""Acquisition portfolio builders and candidate proposal helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from core import normalize_acq_name
from engine.modeling import (
    ModelBundle,
    compute_hv,
    predict_constraint_feasibility,
    weighted_obj_score,
)


@dataclass
class AcqState:
    x_obs: Any
    y_obj: Any
    y_con: Any
    bounds: Any
    reference_point: Any
    q: int = 1
    mc_samples: int = 128
    raw_samples: int = 256
    num_restarts: int = 8
    beta: float = 0.2


def _candidate_output_matrix(values: Any) -> Any:
    """Normalize posterior outputs to shape (num_candidates, num_outputs)."""
    if int(values.ndim) == 0:
        return values.reshape(1, 1)
    if int(values.ndim) == 1:
        return values.reshape(1, int(values.shape[-1]))
    return values.reshape(-1, int(values.shape[-1]))


def _objective_uncertainty_score(variance: Any) -> float:
    std = _candidate_output_matrix(variance).clamp_min(1e-12).sqrt()
    return float(std.mean().detach().cpu().item())


def _normalized_nn_distance(candidate: Any, x_obs: Any, bounds: Any) -> float:
    import torch

    if int(x_obs.shape[0]) == 0:
        return 0.0
    span = (bounds[1] - bounds[0]).clamp_min(1e-12)
    cand_norm = (candidate - bounds[0]) / span
    obs_norm = (x_obs - bounds[0]) / span
    dists = torch.cdist(cand_norm, obs_norm)
    return float(dists.min(dim=-1).values.mean().detach().cpu().item())


def _posterior_mean_hv_proxy(
    y_obj: Any,
    y_con: Any,
    reference_point: Any,
    candidate_mu_obj: Any,
    pred_feas_prob: float,
) -> float:
    import torch

    current_hv = compute_hv(y_obj, y_con, reference_point)
    candidate_mu = _candidate_output_matrix(candidate_mu_obj)
    q = int(candidate_mu.shape[0])
    pseudo_con = -torch.ones(
        q,
        int(y_con.shape[-1]),
        device=y_con.device,
        dtype=y_con.dtype,
    )
    hv_aug = compute_hv(
        torch.cat([y_obj, candidate_mu], dim=0),
        torch.cat([y_con, pseudo_con], dim=0),
        reference_point,
    )
    hv_delta = max(0.0, float(hv_aug - current_hv))
    return float(hv_delta * max(0.0, float(pred_feas_prob)))


def _feasible_objective_partition_points(y_obj: Any, y_con: Any) -> Any:
    feasible = (y_con <= 0.0).all(dim=-1)
    return y_obj[feasible]


def _constraints_for_outputs(num_con: int, offset: int = 0) -> list[Callable[[Any], Any]]:
    funcs = []
    for j in range(num_con):
        idx = int(offset + j)

        def _f(samples: Any, _idx: int = idx) -> Any:
            return samples[..., _idx]

        funcs.append(_f)
    return funcs


def _build_qehvi_like(bundle: ModelBundle, state: AcqState) -> tuple[Any, bool]:
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
    )
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning,
    )

    partition_y = _feasible_objective_partition_points(state.y_obj, state.y_con)
    partitioning = NondominatedPartitioning(
        ref_point=state.reference_point,
        Y=partition_y,
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([state.mc_samples]))
    return (
        qLogExpectedHypervolumeImprovement(
            model=bundle.model_obj,
            ref_point=state.reference_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        ),
        True,
    )


def _build_qlogehvi(bundle: ModelBundle, state: AcqState) -> Any:
    acqf, _ = _build_qehvi_like(bundle, state)
    return acqf


def _build_qlognparego(bundle: ModelBundle, state: AcqState) -> Any:
    import torch
    from botorch.acquisition.multi_objective.parego import qLogNParEGO
    from botorch.sampling.normal import SobolQMCNormalSampler

    m = state.y_obj.shape[-1]
    w = torch.rand(m, device=state.x_obs.device, dtype=state.x_obs.dtype)
    w = w / w.sum().clamp_min(1e-9)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([state.mc_samples]))
    return qLogNParEGO(
        model=bundle.model_obj,
        X_baseline=state.x_obs,
        scalarization_weights=w,
        sampler=sampler,
    )


def _build_qucb(bundle: ModelBundle, state: AcqState) -> Any:
    import torch
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.sampling.normal import SobolQMCNormalSampler

    m = state.y_obj.shape[-1]
    w = torch.ones(m, device=state.x_obs.device, dtype=state.x_obs.dtype) / float(m)
    obj = GenericMCObjective(lambda samples, X=None: (samples * w).sum(dim=-1))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([state.mc_samples]))
    return qUpperConfidenceBound(
        model=bundle.model_obj,
        beta=float(state.beta),
        sampler=sampler,
        objective=obj,
    )


def _build_qlogpof(bundle: ModelBundle, state: AcqState) -> Any:
    import torch
    from botorch.acquisition.logei import qLogProbabilityOfFeasibility
    from botorch.sampling.normal import SobolQMCNormalSampler

    constraints = _constraints_for_outputs(num_con=int(state.y_con.shape[-1]), offset=0)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([state.mc_samples]))
    return qLogProbabilityOfFeasibility(
        model=bundle.model_con,
        constraints=constraints,
        sampler=sampler,
    )


def build_acquisition(acq_name: str, bundle: ModelBundle, state: AcqState) -> Any:
    key = normalize_acq_name(acq_name)
    if key == "qlogehvi":
        return _build_qlogehvi(bundle, state)
    if key == "qlognparego":
        return _build_qlognparego(bundle, state)
    if key == "qucb":
        return _build_qucb(bundle, state)
    if key == "qlogpof":
        return _build_qlogpof(bundle, state)
    raise ValueError(f"Unknown acquisition: {acq_name}")


def propose_candidate(acq_name: str, bundle: ModelBundle, state: AcqState) -> Dict[str, Any]:
    """Optimize one acquisition and return candidate + diagnostics."""
    import torch
    from botorch.optim import optimize_acqf

    acqf = build_acquisition(acq_name, bundle, state)
    candidate, value = optimize_acqf(
        acq_function=acqf,
        bounds=state.bounds,
        q=int(state.q),
        num_restarts=int(state.num_restarts),
        raw_samples=int(state.raw_samples),
        options={"batch_limit": 4, "maxiter": 200},
    )

    # Diagnostics for policy.
    with torch.no_grad():
        y_post = bundle.model_obj.posterior(candidate)
        mu_obj = _candidate_output_matrix(y_post.mean)
        obj_score = weighted_obj_score(mu_obj).mean()
        p_feas, _, _ = predict_constraint_feasibility(bundle.model_con, candidate)
        p_feas_mean = p_feas.mean()
        candidate_uncertainty = _objective_uncertainty_score(y_post.variance)
        candidate_novelty = _normalized_nn_distance(candidate, state.x_obs, state.bounds)
        hv_proxy = _posterior_mean_hv_proxy(
            y_obj=state.y_obj,
            y_con=state.y_con,
            reference_point=state.reference_point,
            candidate_mu_obj=mu_obj,
            pred_feas_prob=float(p_feas_mean.detach().cpu().item()),
        )

    value_f = (
        float(value.detach().cpu().view(-1)[0].item()) if hasattr(value, "detach") else float(value)
    )
    return {
        "acq": acq_name,
        "candidate": candidate.detach(),
        "acq_value": value_f,
        "pred_feas_prob": float(p_feas_mean.detach().cpu().item()),
        "pred_obj_score": float(obj_score.detach().cpu().item()),
        "candidate_uncertainty": float(candidate_uncertainty),
        "candidate_novelty": float(candidate_novelty),
        "posterior_mean_hv_proxy": float(hv_proxy),
    }
