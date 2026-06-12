#!/usr/bin/env python3
"""Run one benchmark/policy configuration across a seed set."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmarks import BENCHMARK_REGISTRY
from core import POLICY_NAMES, ensure_dir, normalize_policy_name, parse_seed_list
from engine.run_seed import SeedRunConfig, run_single_seed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run constrained MO BO campaign for one benchmark/policy."
    )
    p.add_argument("--benchmark", choices=sorted(BENCHMARK_REGISTRY), required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--seeds", default="1-5")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--n-init", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--llm-model", default="gpt-4o")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-timeout", type=int, default=90)
    p.add_argument("--memory-window", type=int, default=10)
    p.add_argument("--env-path", default=".env")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--acq-mc-samples", type=int, default=128)
    p.add_argument("--acq-raw-samples", type=int, default=256)
    p.add_argument("--acq-num-restarts", type=int, default=8)
    p.add_argument("--acq-beta", type=float, default=0.2)
    p.add_argument("--bandit-c", type=float, default=1.0)
    p.add_argument("--bandit-w-feas", type=float, default=0.7)
    p.add_argument("--bandit-w-hv", type=float, default=0.3)
    p.add_argument("--bandit-warmstart", type=int, default=1)
    return p


def _build_seed_cfg(args: argparse.Namespace, seed: int) -> SeedRunConfig:
    return SeedRunConfig(
        benchmark=args.benchmark,
        policy=normalize_policy_name(args.policy),
        seed=int(seed),
        iterations=int(args.iterations),
        q=int(args.q),
        n_init=int(args.n_init),
        results_dir=str(args.results_dir),
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        llm_timeout=int(args.llm_timeout),
        memory_window=int(args.memory_window),
        env_path=args.env_path,
        device=args.device,
        dtype=args.dtype,
        acq_mc_samples=int(args.acq_mc_samples),
        acq_raw_samples=int(args.acq_raw_samples),
        acq_num_restarts=int(args.acq_num_restarts),
        acq_beta=float(args.acq_beta),
        bandit_c=float(args.bandit_c),
        bandit_w_feas=float(args.bandit_w_feas),
        bandit_w_hv=float(args.bandit_w_hv),
        bandit_warmstart=int(args.bandit_warmstart),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    args.policy = normalize_policy_name(args.policy)
    if args.policy not in set(POLICY_NAMES):
        raise SystemExit(f"Invalid --policy '{args.policy}'. Valid: {sorted(POLICY_NAMES)}")
    seeds = parse_seed_list(args.seeds)
    out_dir = ensure_dir(args.results_dir)
    cfg_dump = vars(args).copy()
    cfg_dump["seeds_parsed"] = seeds
    (out_dir / "run_config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")

    t0 = time.time()
    workers = max(1, int(args.workers))
    print(
        f"[run_campaign] benchmark={args.benchmark} policy={args.policy} "
        f"seeds={len(seeds)} iterations={args.iterations} workers={workers}"
    )

    if workers == 1:
        for seed in seeds:
            run_single_seed(_build_seed_cfg(args, seed))
            print(f"[run_campaign] seed={seed} done")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_single_seed, _build_seed_cfg(args, s)) for s in seeds]
            for fut in as_completed(futs):
                seed_done = fut.result()
                print(f"[run_campaign] seed={seed_done} done")

    print(f"[run_campaign] done in {time.time() - t0:.2f}s -> {out_dir}")


if __name__ == "__main__":
    main()
