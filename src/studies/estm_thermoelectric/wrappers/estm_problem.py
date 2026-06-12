"""Domain definition and observation protocol for the ESTM study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch

from engine.modeling import compute_hv
from studies.estm_thermoelectric.constants import (
    CONSTRAINT_COLUMNS,
    ESTM_DATA_PATH,
    ESTM_MANIFEST_PATH,
    FEATURE_COLUMNS,
    OBJECTIVE_COLUMNS,
)


@dataclass
class ESTMProblem:
    name: str
    df: pd.DataFrame
    x_all: Any
    y_obj_all: Any
    y_con_all: Any
    reference_point: Any
    fixed_ranges: Any
    bounds: Any
    feature_columns: tuple[str, ...]
    objective_columns: tuple[str, ...]
    constraint_columns: tuple[str, ...]
    manifest: Dict[str, Any]

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


def _as_tensor(
    df: pd.DataFrame, columns: Iterable[str], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    arr = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    if pd.isna(arr).any():
        raise ValueError(f"ESTM dataset has NaNs in columns: {list(columns)}")
    return torch.tensor(arr, device=device, dtype=dtype)


def _load_manifest(path: str | None = None) -> Dict[str, Any]:
    manifest_path = ESTM_MANIFEST_PATH if path is None else Path(path)
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_problem(config: Any, device: str = "auto", dtype: str = "float64") -> ESTMProblem:
    data_path = _cfg_value(config, "data_path", str(ESTM_DATA_PATH))
    manifest_path = _cfg_value(config, "manifest_path", str(ESTM_MANIFEST_PATH))
    df = pd.read_csv(data_path)
    manifest = _load_manifest(manifest_path)
    device_t = _resolve_device(device)
    dtype_t = _resolve_dtype(dtype)

    x_all = _as_tensor(df, FEATURE_COLUMNS, device=device_t, dtype=dtype_t)
    y_obj_all = _as_tensor(df, OBJECTIVE_COLUMNS, device=device_t, dtype=dtype_t)
    y_con_all = _as_tensor(df, CONSTRAINT_COLUMNS, device=device_t, dtype=dtype_t)
    reference_point = torch.zeros(int(y_obj_all.shape[-1]), device=device_t, dtype=dtype_t)
    fixed_ranges = torch.ones(int(y_obj_all.shape[-1]), device=device_t, dtype=dtype_t)
    bounds = torch.stack([x_all.min(dim=0).values, x_all.max(dim=0).values], dim=0)

    return ESTMProblem(
        name="estm_thermoelectric",
        df=df,
        x_all=x_all,
        y_obj_all=y_obj_all,
        y_con_all=y_con_all,
        reference_point=reference_point,
        fixed_ranges=fixed_ranges,
        bounds=bounds,
        feature_columns=FEATURE_COLUMNS,
        objective_columns=OBJECTIVE_COLUMNS,
        constraint_columns=CONSTRAINT_COLUMNS,
        manifest=manifest,
    )


def initial_seed_selection(problem: ESTMProblem, seed: int, n_init: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    total = int(problem.x_all.shape[0])
    n = max(1, min(int(n_init), total))
    all_idx = torch.arange(total, dtype=torch.long)
    perm = torch.randperm(total, generator=generator)
    return all_idx[perm[:n]]


def evaluate_selected_candidate(problem: ESTMProblem, selected_index: int) -> Dict[str, Any]:
    idx = int(selected_index)
    observed_feasible = bool((problem.y_con_all[idx] <= 0.0).all().item())
    row = problem.df.iloc[idx]
    return {
        "selected_index": idx,
        "observed_feasible": int(observed_feasible),
        "full_measurement_observed": True,
        "y_obj_observed": problem.y_obj_all[idx].detach().clone(),
        "y_con_observed": problem.y_con_all[idx].detach().clone(),
        "x_selected": problem.x_all[idx].detach().clone(),
        "formula": str(row.get("formula", "")),
        "temperature_k": float(row.get("temperature_k", 0.0)),
        "source_reference": str(row.get("source_reference", "")),
    }


def compute_hv_and_feasible_metrics(
    problem: ESTMProblem, observed_indices: Iterable[int]
) -> Dict[str, Any]:
    indices = [int(idx) for idx in observed_indices]
    if not indices:
        return {
            "hypervolume_raw": 0.0,
            "hypervolume_scaled_fixed_range": 0.0,
            "true_pareto_count": 0,
            "cumulative_feasible_count": 0,
            "cumulative_observed_count": 0,
        }
    idx_t = torch.tensor(indices, device=problem.x_all.device, dtype=torch.long)
    y_obj = problem.y_obj_all[idx_t]
    y_con = problem.y_con_all[idx_t]
    feasible_mask = (y_con <= 0.0).all(dim=-1)
    hv = compute_hv(y_obj, y_con, problem.reference_point)
    if not bool(feasible_mask.any()):
        true_pareto_count = 0
    else:
        front = y_obj[feasible_mask]
        keep = torch.ones(int(front.shape[0]), dtype=torch.bool, device=front.device)
        for i in range(int(front.shape[0])):
            if not bool(keep[i]):
                continue
            dominates = (front >= front[i]).all(dim=-1) & (front > front[i]).any(dim=-1)
            dominates[i] = False
            if bool(dominates.any()):
                keep[i] = False
        true_pareto_count = int(keep.sum().item())
    return {
        "hypervolume_raw": float(hv),
        "hypervolume_scaled_fixed_range": float(hv),
        "true_pareto_count": int(true_pareto_count),
        "cumulative_feasible_count": int(feasible_mask.sum().item()),
        "cumulative_observed_count": int(len(indices)),
    }
