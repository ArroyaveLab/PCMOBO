"""Benchmark adapters with a unified constrained MO interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class BenchmarkAdapter:
    """Unified adapter interface used by the BO engine."""

    name: str
    dim: int
    num_objectives: int
    num_constraints: int
    bounds: Any
    reference_point: Any
    problem: Any

    def evaluate(self, x: Any) -> Tuple[Any, Any]:
        """Return (objectives_maximize, constraints_leq_zero)."""
        problem_bounds = getattr(self.problem, "bounds", None)
        target_device = getattr(problem_bounds, "device", x.device)
        target_dtype = getattr(problem_bounds, "dtype", x.dtype)
        x_eval = x.to(device=target_device, dtype=target_dtype)
        y_raw = None
        if hasattr(self.problem, "evaluate_true"):
            y_raw = self.problem.evaluate_true(x_eval)
        elif callable(self.problem):
            y_raw = self.problem(x_eval)
        else:
            raise RuntimeError(f"{self.name}: benchmark object is not evaluable.")

        # Convert objectives to maximize form.
        y = -y_raw

        if hasattr(self.problem, "evaluate_slack_true"):
            slack = self.problem.evaluate_slack_true(x_eval)
            c = -slack
        elif hasattr(self.problem, "evaluate_slack"):
            slack = self.problem.evaluate_slack(x_eval)
            c = -slack
        else:
            raise RuntimeError(f"{self.name}: benchmark has no constraint-evaluation method.")

        return y.to(device=x.device, dtype=x.dtype), c.to(device=x.device, dtype=x.dtype)


BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "constrained_branin_currin": {
        "cls_name": "ConstrainedBraninCurrin",
        "dim": 2,
        "num_objectives": 2,
        "num_constraints": 1,
        "raw_reference_point": [80.0, 12.0],
    },
    "constr": {
        "cls_name": "CONSTR",
        "dim": 2,
        "num_objectives": 2,
        "num_constraints": 2,
        "raw_reference_point": [10.0, 10.0],
    },
    "c2dtlz2_2obj": {
        "cls_name": "C2DTLZ2",
        "dim": 11,
        "num_objectives": 2,
        "num_constraints": 1,
        "raw_reference_point": [1.1, 1.1],
    },
    "c2dtlz2_3obj": {
        "cls_name": "C2DTLZ2",
        "dim": 12,
        "num_objectives": 3,
        "num_constraints": 1,
        "raw_reference_point": [1.1, 1.1, 1.1],
    },
    "disc_brake": {
        "cls_name": "DiscBrake",
        "dim": 4,
        "num_objectives": 2,
        "num_constraints": 4,
        "raw_reference_point": [5.7771, 3.9651],
    },
}


def _load_test_function_classes() -> Dict[str, Any]:
    try:
        from botorch.test_functions.multi_objective import (
            C2DTLZ2,
            CONSTR,
            ConstrainedBraninCurrin,
            DiscBrake,
        )  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "BoTorch test functions are unavailable. Install torch/botorch/gpytorch."
        ) from exc
    return {
        "ConstrainedBraninCurrin": ConstrainedBraninCurrin,
        "CONSTR": CONSTR,
        "C2DTLZ2": C2DTLZ2,
        "DiscBrake": DiscBrake,
    }


def _instantiate_problem(cls: Any, dim: int, num_objectives: int) -> Any:
    if cls.__name__ == "C2DTLZ2":
        return cls(dim=dim, num_objectives=num_objectives, negate=False)
    return cls(negate=False)


def _maximize_space_reference_point(entry: Dict[str, Any], device: Any, dtype: Any) -> Any:
    import torch

    raw_reference_point = entry.get("raw_reference_point")
    if raw_reference_point is None:
        raise RuntimeError(f"Benchmark '{entry}' is missing a raw_reference_point.")
    if len(raw_reference_point) != int(entry["num_objectives"]):
        raise RuntimeError(
            "Benchmark reference point length does not match the number of objectives."
        )
    # BoTorch benchmark definitions are minimization-space by default.
    # PCMOBO negates objectives into maximize space, so negate the raw point as well.
    return -torch.tensor(raw_reference_point, device=device, dtype=dtype)


def build_benchmark(name: str, device: Any, dtype: Any) -> BenchmarkAdapter:
    key = str(name).strip().lower()
    if key not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark '{name}'. Valid: {sorted(BENCHMARK_REGISTRY)}")

    import torch

    entry = BENCHMARK_REGISTRY[key]
    classes = _load_test_function_classes()
    cls = classes[entry["cls_name"]]
    problem = _instantiate_problem(
        cls=cls,
        dim=int(entry["dim"]),
        num_objectives=int(entry["num_objectives"]),
    )
    if hasattr(problem, "to"):
        problem = problem.to(device=device, dtype=dtype)

    if hasattr(problem, "bounds"):
        bounds = problem.bounds.to(device=device, dtype=dtype)
    else:
        bounds = torch.stack(
            [
                torch.zeros(int(entry["dim"]), device=device, dtype=dtype),
                torch.ones(int(entry["dim"]), device=device, dtype=dtype),
            ]
        )

    adapter = BenchmarkAdapter(
        name=key,
        dim=int(entry["dim"]),
        num_objectives=int(entry["num_objectives"]),
        num_constraints=int(entry["num_constraints"]),
        bounds=bounds,
        reference_point=None,
        problem=problem,
    )
    adapter.reference_point = _maximize_space_reference_point(
        entry=entry,
        device=device,
        dtype=dtype,
    )
    return adapter
