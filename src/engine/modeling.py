"""Model fitting and prediction helpers for constrained MO BO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class ModelBundle:
    model_obj: Any
    model_con: Any
    model_all: Any


def _posterior_output_matrix(values: Any) -> Any:
    """Normalize posterior outputs to shape (num_candidates, num_outputs)."""
    if int(values.ndim) == 0:
        return values.reshape(1, 1)
    if int(values.ndim) == 1:
        return values.reshape(1, int(values.shape[-1]))
    return values.reshape(-1, int(values.shape[-1]))


def _outcome_transform_for(y: Any, standardize_cls: Any) -> Any:
    import torch

    if int(y.shape[0]) < 2:
        return None
    std = torch.std(y, dim=0)
    if bool((std < 1e-12).all()):
        return None
    return standardize_cls(m=int(y.shape[-1]))


def fit_output_model(x: Any, y: Any) -> Any:
    """Fit independent single-task GPs for one or more outputs."""
    try:
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("Missing BoTorch/GPyTorch dependencies.") from exc

    d = x.shape[-1]
    outputs = []
    for i in range(y.shape[-1]):
        yi = y[:, i : i + 1]
        outputs.append(
            SingleTaskGP(
                train_X=x,
                train_Y=yi,
                input_transform=Normalize(d=d),
                outcome_transform=_outcome_transform_for(yi, Standardize),
            )
        )

    model = ModelListGP(*outputs)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def fit_model_bundle(x: Any, y_obj: Any, y_con: Any) -> ModelBundle:
    """Fit independent single-task GPs for objective and constraint outputs."""
    try:
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("Missing BoTorch/GPyTorch dependencies.") from exc

    d = x.shape[-1]
    m = y_obj.shape[-1]
    k = y_con.shape[-1]

    obj_models = []
    for i in range(m):
        yi = y_obj[:, i : i + 1]
        obj_models.append(
            SingleTaskGP(
                train_X=x,
                train_Y=yi,
                input_transform=Normalize(d=d),
                outcome_transform=_outcome_transform_for(yi, Standardize),
            )
        )

    con_models = []
    for j in range(k):
        cj = y_con[:, j : j + 1]
        con_models.append(
            SingleTaskGP(
                train_X=x,
                train_Y=cj,
                input_transform=Normalize(d=d),
                outcome_transform=_outcome_transform_for(cj, Standardize),
            )
        )

    model_obj = ModelListGP(*obj_models)
    model_con = ModelListGP(*con_models)
    model_all = ModelListGP(*(obj_models + con_models))
    mll = SumMarginalLogLikelihood(model_all.likelihood, model_all)
    fit_gpytorch_mll(mll)
    model_obj.eval()
    model_con.eval()
    model_all.eval()
    return ModelBundle(model_obj=model_obj, model_con=model_con, model_all=model_all)


def predict_constraint_feasibility(
    model_con: Any, x: Any, eps: float = 1e-9
) -> Tuple[Any, Any, Any]:
    """Return (p_feasible, mu, sigma) under independence across constraints."""
    import torch

    post = model_con.posterior(x)
    mu = _posterior_output_matrix(post.mean)
    var = _posterior_output_matrix(post.variance).clamp_min(eps)
    sigma = var.sqrt()
    z = (0.0 - mu) / sigma
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
    )
    p_each = normal.cdf(z).clamp(1e-9, 1.0)
    p_all = p_each.prod(dim=-1)
    return p_all, mu, sigma


def compute_hv(y_obj: Any, y_con: Any, reference_point: Any) -> float:
    """Hypervolume over feasible objective points in maximize space."""

    feasible = (y_con <= 0.0).all(dim=-1)
    if not bool(feasible.any()):
        return 0.0
    y_feas = y_obj[feasible]
    try:
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from botorch.utils.multi_objective.pareto import is_non_dominated
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("BoTorch hypervolume utilities are unavailable.") from exc

    nd_mask = is_non_dominated(y_feas)
    front = y_feas[nd_mask]
    hv = Hypervolume(ref_point=reference_point)
    value = hv.compute(front)
    return float(value.detach().cpu().item() if hasattr(value, "detach") else value)


def weighted_obj_score(y: Any) -> Any:
    """Simple normalized weighted objective score for diagnostics."""
    import torch

    m = y.shape[-1]
    w = torch.ones(m, device=y.device, dtype=y.dtype) / float(m)
    return (y * w).sum(dim=-1)
