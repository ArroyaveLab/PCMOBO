"""Domain definition and observation protocol for the HEA study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import pandas as pd
import torch

from engine.modeling import compute_hv
from studies.hea_design.constants import (
    HEA_DATA_PATH,
    INPUT_COLUMNS,
    OBJECTIVE_COLUMNS,
    RAW_THRESHOLDS,
    SCALED_THRESHOLDS,
)


@dataclass
class HEAProblem:
    name: str
    df: pd.DataFrame
    x_all: Any
    y_obj_all: Any
    y_con_obj_all: Any
    y_phase_con_all: Any
    y_con_all: Any
    reference_point: Any
    fixed_ranges: Any
    bounds: Any
    thresholds: Dict[str, float]
    threshold_mode: str
    fixed_range_scope: str
    input_columns: tuple[str, ...]
    bcc_columns: tuple[str, ...]
    bcc_single_mask: Any

    @property
    def dim(self) -> int:
        return int(self.x_all.shape[-1])

    @property
    def num_objectives(self) -> int:
        return int(self.y_obj_all.shape[-1])

    @property
    def num_constraints(self) -> int:
        return int(self.y_con_all.shape[-1])


def _cfg_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_device(device: str) -> torch.device:
    raw = str(device).strip().lower()
    if raw in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _resolve_dtype(dtype: str) -> torch.dtype:
    raw = str(dtype).strip().lower()
    if raw == "float32":
        return torch.float32
    return torch.float64


def _phase_columns(df: pd.DataFrame) -> tuple[str, ...]:
    cols = tuple(c for c in df.columns if "600C" in str(c) and "BCC" in str(c))
    if not cols:
        raise ValueError("HEA dataset is missing 600C/BCC phase columns.")
    return cols


def _threshold_mode(df: pd.DataFrame, requested: str) -> str:
    raw = str(requested or "auto").strip().lower()
    if raw not in {"auto", "scaled", "raw"}:
        raise ValueError(f"Unsupported threshold mode: {requested}")
    if raw != "auto":
        return raw
    density_max = float(pd.to_numeric(df["PROP 25C Density (g/cm3)"], errors="coerce").max())
    return "scaled" if density_max <= 1.5 else "raw"


def _thresholds(df: pd.DataFrame, config: Any) -> tuple[str, Dict[str, float]]:
    mode = _threshold_mode(df, _cfg_value(config, "threshold_mode", "auto"))
    defaults = SCALED_THRESHOLDS if mode == "scaled" else RAW_THRESHOLDS
    density_override = _cfg_value(config, "density_thresh", None)
    ys_override = _cfg_value(config, "ys_thresh", None)
    pugh_override = _cfg_value(config, "pugh_thresh", None)
    st_override = _cfg_value(config, "st_thresh", None)
    out = {
        "density": float(defaults["density"] if density_override is None else density_override),
        "ys": float(defaults["ys"] if ys_override is None else ys_override),
        "pugh": float(defaults["pugh"] if pugh_override is None else pugh_override),
        "st": float(defaults["st"] if st_override is None else st_override),
    }
    return mode, out


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = [col for col in INPUT_COLUMNS + OBJECTIVE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"HEA dataset missing required columns: {missing}")
    if "VEC" not in df.columns and "VEC Avg" not in df.columns:
        raise ValueError("HEA dataset must include either 'VEC' or 'VEC Avg'.")


def _as_tensor(
    df: pd.DataFrame, columns: Iterable[str], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    arr = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    if pd.isna(arr).any():
        raise ValueError(f"HEA dataset has NaNs in columns: {list(columns)}")
    return torch.tensor(arr, device=device, dtype=dtype)


def _reference_point(y_obj: torch.Tensor) -> torch.Tensor:
    lo = y_obj.min(dim=0).values
    hi = y_obj.max(dim=0).values
    span = (hi - lo).clamp_min(1e-6)
    return lo - 0.1 * span


def _fixed_ranges(y_obj: torch.Tensor) -> torch.Tensor:
    lo = y_obj.min(dim=0).values
    hi = y_obj.max(dim=0).values
    return (hi - lo).clamp_min(1e-6)


def _bcc_single_tensor(
    df: pd.DataFrame, bcc_columns: tuple[str, ...], *, device: torch.device
) -> torch.Tensor:
    total = (
        df.loc[:, list(bcc_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
        .to_numpy(dtype="float64")
    )
    mask = total >= 0.99
    return torch.tensor(mask, device=device, dtype=torch.bool)


def _prepare_dataframe(df: pd.DataFrame, bcc_columns: tuple[str, ...]) -> pd.DataFrame:
    working = df.copy()
    working["_bcc_total"] = (
        working.loc[:, list(bcc_columns)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    )
    required = list(INPUT_COLUMNS + OBJECTIVE_COLUMNS)
    cleaned = working.dropna(subset=required).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(
            "HEA dataset has no rows remaining after dropping NaNs in required columns."
        )
    return cleaned


def load_problem(config: Any, device: str = "auto", dtype: str = "float64") -> HEAProblem:
    data_path = _cfg_value(config, "data_path", str(HEA_DATA_PATH))
    raw_df = pd.read_csv(data_path)
    _ensure_columns(raw_df)
    phase_cols = _phase_columns(raw_df)
    df = _prepare_dataframe(raw_df, phase_cols)
    device_t = _resolve_device(device)
    dtype_t = _resolve_dtype(dtype)
    threshold_mode, thresholds = _thresholds(df, config)
    fixed_range_scope = str(_cfg_value(config, "fixed_range_scope", "all")).strip().lower()
    if fixed_range_scope not in {"all", "bcc_only"}:
        raise ValueError(f"Unsupported fixed range scope: {fixed_range_scope}")

    x_all = _as_tensor(df, INPUT_COLUMNS, device=device_t, dtype=dtype_t)
    st = _as_tensor(df, ["PROP ST (K)"], device=device_t, dtype=dtype_t)
    density = _as_tensor(df, ["PROP 25C Density (g/cm3)"], device=device_t, dtype=dtype_t)
    ys = _as_tensor(df, ["YS 600 C PRIOR"], device=device_t, dtype=dtype_t)
    pugh = _as_tensor(df, ["Pugh_Ratio_PRIOR"], device=device_t, dtype=dtype_t)
    bcc_single_mask = _bcc_single_tensor(df, phase_cols, device=device_t)

    y_obj_all = torch.cat([st, -density, ys, pugh], dim=-1)
    c_density = density - float(thresholds["density"])
    c_ys = float(thresholds["ys"]) - ys
    c_pugh = float(thresholds["pugh"]) - pugh
    c_st = float(thresholds["st"]) - st
    y_con_obj_all = torch.cat([c_density, c_ys, c_pugh, c_st], dim=-1)
    y_phase_con_all = torch.where(
        bcc_single_mask.unsqueeze(-1),
        torch.full((len(df), 1), -1.0, device=device_t, dtype=dtype_t),
        torch.full((len(df), 1), 1.0, device=device_t, dtype=dtype_t),
    )
    y_con_all = torch.cat([y_con_obj_all, y_phase_con_all], dim=-1)

    feasible_mask = (y_con_all <= 0.0).all(dim=-1)
    reference_source = y_obj_all[feasible_mask] if bool(feasible_mask.any()) else y_obj_all
    reference_point = _reference_point(reference_source)

    range_source = y_obj_all
    if fixed_range_scope == "bcc_only" and bool(bcc_single_mask.any()):
        range_source = y_obj_all[bcc_single_mask]
    fixed_ranges = _fixed_ranges(range_source)

    bounds = torch.stack([x_all.min(dim=0).values, x_all.max(dim=0).values], dim=0)
    return HEAProblem(
        name="hea_design",
        df=df,
        x_all=x_all,
        y_obj_all=y_obj_all,
        y_con_obj_all=y_con_obj_all,
        y_phase_con_all=y_phase_con_all,
        y_con_all=y_con_all,
        reference_point=reference_point,
        fixed_ranges=fixed_ranges,
        bounds=bounds,
        thresholds=thresholds,
        threshold_mode=threshold_mode,
        fixed_range_scope=fixed_range_scope,
        input_columns=INPUT_COLUMNS,
        bcc_columns=phase_cols,
        bcc_single_mask=bcc_single_mask,
    )


def initial_seed_selection(problem: HEAProblem, seed: int, n_init: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    total = int(problem.x_all.shape[0])
    n = max(1, min(int(n_init), total))
    bcc_idx = torch.nonzero(problem.bcc_single_mask.detach().cpu(), as_tuple=False).view(-1)
    if int(bcc_idx.numel()) >= n:
        perm = torch.randperm(int(bcc_idx.numel()), generator=generator)
        return bcc_idx[perm[:n]]
    all_idx = torch.arange(total, dtype=torch.long)
    remaining_mask = torch.ones(total, dtype=torch.bool)
    remaining_mask[bcc_idx] = False
    remaining = all_idx[remaining_mask]
    need = n - int(bcc_idx.numel())
    if int(remaining.numel()) > 0:
        perm = torch.randperm(int(remaining.numel()), generator=generator)
        remainder = remaining[perm[:need]]
        chosen = torch.cat([bcc_idx, remainder], dim=0)
    else:
        chosen = bcc_idx
    return chosen


def evaluate_selected_candidate(problem: HEAProblem, selected_index: int) -> Dict[str, Any]:
    idx = int(selected_index)
    observed_bcc_single = bool(problem.bcc_single_mask[idx].item())
    full_measurement_observed = observed_bcc_single
    observed_feasible = False
    y_obj_observed = None
    y_con_obj_observed = None
    if full_measurement_observed:
        y_obj_observed = problem.y_obj_all[idx].detach().clone()
        y_con_obj_observed = problem.y_con_obj_all[idx].detach().clone()
        observed_feasible = bool((problem.y_con_all[idx] <= 0.0).all().item())
    return {
        "selected_index": idx,
        "observed_bcc_single": observed_bcc_single,
        "observed_feasible": int(observed_feasible),
        "full_measurement_observed": full_measurement_observed,
        "y_obj_observed": y_obj_observed,
        "y_con_obj_observed": y_con_obj_observed,
        "y_phase_observed": problem.y_phase_con_all[idx].detach().clone(),
        "x_selected": problem.x_all[idx].detach().clone(),
    }


def _scaled_hv(problem: HEAProblem, y_obj: torch.Tensor, y_con: torch.Tensor) -> float:
    scaled_obj = (y_obj - problem.reference_point) / problem.fixed_ranges.clamp_min(1e-9)
    scaled_ref = torch.zeros_like(problem.reference_point)
    return compute_hv(scaled_obj, y_con, scaled_ref)


def _true_pareto_count(y_obj: torch.Tensor, y_con: torch.Tensor) -> int:
    feasible = (y_con <= 0.0).all(dim=-1)
    if not bool(feasible.any()):
        return 0
    front = y_obj[feasible]
    keep = torch.ones(int(front.shape[0]), dtype=torch.bool, device=front.device)
    for i in range(int(front.shape[0])):
        if not bool(keep[i]):
            continue
        dominates = (front >= front[i]).all(dim=-1) & (front > front[i]).any(dim=-1)
        dominates[i] = False
        if bool(dominates.any()):
            keep[i] = False
    return int(keep.sum().item())


def compute_hv_and_feasible_metrics(
    problem: HEAProblem, observed_full_indices: Iterable[int]
) -> Dict[str, Any]:
    indices = [int(idx) for idx in observed_full_indices]
    if not indices:
        return {
            "hypervolume_raw": 0.0,
            "hypervolume_scaled_fixed_range": 0.0,
            "true_pareto_count": 0,
            "cumulative_feasible_count": 0,
        }
    idx_t = torch.tensor(indices, device=problem.x_all.device, dtype=torch.long)
    y_obj = problem.y_obj_all[idx_t]
    y_con = problem.y_con_all[idx_t]
    feasible_mask = (y_con <= 0.0).all(dim=-1)
    return {
        "hypervolume_raw": compute_hv(y_obj, y_con, problem.reference_point),
        "hypervolume_scaled_fixed_range": _scaled_hv(problem, y_obj, y_con),
        "true_pareto_count": _true_pareto_count(y_obj, y_con),
        "cumulative_feasible_count": int(feasible_mask.sum().item()),
    }
