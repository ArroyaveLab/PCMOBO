"""LLM policy for acquisition-portfolio switching."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from core import compact_json, safe_float
from llm_runtime import LangGraphJSONClient

ALLOWED_ACQS = ("qlogehvi", "qlognparego", "qucb", "qlogpof")
THREE_AGENT_PROMPT_VERSION = "v5_three_agent_portfolio_advocates"
POLICY_PROMPT_VERSION = THREE_AGENT_PROMPT_VERSION
ARM_SEMANTICS = {
    "qlogehvi": "pareto improvement",
    "qlognparego": "scalarized tradeoff",
    "qucb": "uncertainty exploration",
    "qlogpof": "feasibility seeking",
}
FEASIBILITY_ADVOCATE_PROMPT = (
    "You are the Feasibility Advocate in a BO portfolio-switch pipeline for constrained multi-objective "
    "optimization. Choose exactly one acquisition from: qlogehvi, qlognparego, qucb, qlogpof. "
    "Arm semantics: qlogehvi=pareto improvement, qlognparego=scalarized tradeoff, "
    "qucb=uncertainty exploration, qlogpof=feasibility seeking. "
    "Your role is to argue for the arm most justified by feasible discovery and feasibility protection given "
    "the current run state. Use only the evidence in metadata, global_state, per_arm, and recent_history. "
    "recent_history contains compact past decision snapshots with chosen outcomes and the model diagnostics that "
    "were visible at those decision times. Make the strongest feasibility-grounded recommendation you can, "
    "but do not invent data and do not use any action outside metadata.allowed_arms. Return strict JSON only "
    "with keys: recommended_acq, reason, argument, confidence."
)
HV_ADVOCATE_PROMPT = (
    "You are the HV Advocate in a BO portfolio-switch pipeline for constrained multi-objective optimization. "
    "Choose exactly one acquisition from: qlogehvi, qlognparego, qucb, qlogpof. "
    "Arm semantics: qlogehvi=pareto improvement, qlognparego=scalarized tradeoff, "
    "qucb=uncertainty exploration, qlogpof=feasibility seeking. "
    "Your role is to argue for the arm most justified by feasible hypervolume growth over time. Use only the "
    "evidence in metadata, global_state, per_arm, and recent_history. recent_history contains compact past "
    "decision snapshots with chosen outcomes and the model diagnostics that were visible at those decision times. "
    "Make the strongest frontier-growth recommendation you can, but keep the focus on feasible hypervolume, "
    "not raw objective improvement in infeasible regions. Return strict JSON only with keys: "
    "recommended_acq, reason, argument, confidence."
)
ARBITER_PROMPT = (
    "You are the Arbiter in a BO portfolio-switch pipeline for constrained multi-objective optimization. "
    "Choose exactly one acquisition from: qlogehvi, qlognparego, qucb, qlogpof. "
    "Arm semantics: qlogehvi=pareto improvement, qlognparego=scalarized tradeoff, "
    "qucb=uncertainty exploration, qlogpof=feasibility seeking. "
    "Your objective is to choose the single arm that best serves feasible hypervolume over the run while "
    "protecting feasible discovery when needed. You will receive the raw state plus a feasibility-grounded "
    "advocate recommendation and a feasible-HV-grounded advocate recommendation. Treat the advocate outputs as "
    "evidence, not commands. You may agree with either advocate or disagree with both if the raw state supports "
    "a different choice. Use the full context and reason through the tradeoff. The reflection should briefly "
    "cite the main evidence behind the final choice. Return strict JSON only with keys: acq, reason, "
    "reflection, confidence."
)


class _StageInvocationError(RuntimeError):
    def __init__(self, stage_name: str, message: str, *, json_payload: str = "") -> None:
        super().__init__(message)
        self.stage_name = str(stage_name)
        self.json_payload = str(json_payload)


def load_env_file(env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        value = v.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _validate_stage_acq(value: Any) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "ehvi": "qlogehvi",
        "nparego": "qlognparego",
        "ucb": "qucb",
        "pof": "qlogpof",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in ALLOWED_ACQS:
        raise ValueError(f"invalid acquisition '{value}'")
    return normalized


def _global_state(state: Dict[str, Any]) -> Dict[str, Any]:
    raw = state.get("global_state", {})
    return raw if isinstance(raw, dict) else {}


def _per_arm_state(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = state.get("per_arm", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


@dataclass
class LLMSwitchPolicy:
    model: str = "gpt-4o"
    llm_provider: str = "openai"
    env_path: str = ".env"
    timeout_s: int = 90
    temperature: float = 0.1
    memory_window: int = 10

    def __post_init__(self) -> None:
        load_env_file(self.env_path)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.memory_window = max(1, int(self.memory_window))
        self.llm_provider = str(self.llm_provider).strip().lower() or "openai"
        self.client = LangGraphJSONClient(
            model=self.model,
            provider=self.llm_provider,
            env_path=self.env_path,
            timeout_s=self.timeout_s,
            temperature=self.temperature,
        )

    def _build_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metadata = state.get("metadata", {}) if isinstance(state.get("metadata"), dict) else {}
        global_state = _global_state(state)
        per_arm = _per_arm_state(state)
        recent_history = (
            state.get("recent_history", []) if isinstance(state.get("recent_history"), list) else []
        )

        return {
            "metadata": {
                "benchmark": str(metadata.get("benchmark", "")),
                "iteration": int(metadata.get("iteration", 0)),
                "total_iterations": int(metadata.get("total_iterations", 0)),
                "allowed_arms": [str(x) for x in metadata.get("allowed_arms", list(ALLOWED_ACQS))],
            },
            "global_state": {
                "progress_ratio": round(safe_float(global_state.get("progress_ratio", 0.0)), 6),
                "cumulative_feasible_count": int(global_state.get("cumulative_feasible_count", 0)),
                "current_hv": round(safe_float(global_state.get("current_hv", 0.0)), 8),
                "hv_gain_since_init": round(
                    safe_float(global_state.get("hv_gain_since_init", 0.0)), 8
                ),
                "observed_feasible_fraction": round(
                    safe_float(global_state.get("observed_feasible_fraction", 0.0)),
                    6,
                ),
                "recent_feasible_gain_rate": round(
                    safe_float(global_state.get("recent_feasible_gain_rate", 0.0)),
                    6,
                ),
                "recent_hv_gain_mean": round(
                    safe_float(global_state.get("recent_hv_gain_mean", 0.0)), 8
                ),
                "recent_hv_gain_std": round(
                    safe_float(global_state.get("recent_hv_gain_std", 0.0)), 8
                ),
                "feasibility_stage": str(global_state.get("feasibility_stage", "none")),
                "last_selected_acq": str(global_state.get("last_selected_acq", "")),
                "same_acq_streak": int(global_state.get("same_acq_streak", 0) or 0),
                "current_arm_no_hv_gain_streak": int(
                    global_state.get("current_arm_no_hv_gain_streak", 0) or 0
                ),
                "iterations_since_last_switch": int(
                    global_state.get("iterations_since_last_switch", 0) or 0
                ),
            },
            "per_arm": {
                arm: {
                    "semantic_role": ARM_SEMANTICS.get(arm, arm),
                    "predicted_feasibility": round(
                        safe_float(values.get("predicted_feasibility", 0.0)), 6
                    ),
                    "pred_obj_score": round(safe_float(values.get("pred_obj_score", 0.0)), 8),
                    "normalized_acq_value": round(
                        safe_float(values.get("normalized_acq_value", 0.0)), 6
                    ),
                    "posterior_mean_hv_proxy": round(
                        safe_float(values.get("posterior_mean_hv_proxy", 0.0)),
                        8,
                    ),
                    "candidate_uncertainty": round(
                        safe_float(values.get("candidate_uncertainty", 0.0)), 8
                    ),
                    "candidate_novelty": round(safe_float(values.get("candidate_novelty", 0.0)), 8),
                    "recent_count": int(values.get("recent_count", 0) or 0),
                    "recent_feasible_gain_rate": round(
                        safe_float(values.get("recent_feasible_gain_rate", 0.0)),
                        6,
                    ),
                    "recent_hv_gain_mean": round(
                        safe_float(values.get("recent_hv_gain_mean", 0.0)), 8
                    ),
                    "iterations_since_selected": int(
                        values.get("iterations_since_selected", 0) or 0
                    ),
                    "feasibility_rank": int(values.get("feasibility_rank", 0) or 0),
                    "hv_proxy_rank": int(values.get("hv_proxy_rank", 0) or 0),
                    "uncertainty_rank": int(values.get("uncertainty_rank", 0) or 0),
                }
                for arm, values in per_arm.items()
                if arm in ALLOWED_ACQS
            },
            "recent_history": [
                {
                    "iteration": int(item.get("iteration", 0)),
                    "chosen_acq": str(item.get("chosen_acq", item.get("selected_acq", ""))),
                    "observed_feasible": int(item.get("observed_feasible", 0) or 0),
                    "feasible_gain": int(item.get("feasible_gain", 0) or 0),
                    "hv_gain": round(safe_float(item.get("hv_gain", 0.0)), 8),
                    "cumulative_feasible_count": int(item.get("cumulative_feasible_count", 0) or 0),
                    "feasibility_stage_at_decision": str(
                        item.get("feasibility_stage_at_decision", "none")
                    ),
                    "same_acq_streak_at_decision": int(
                        item.get("same_acq_streak_at_decision", 0) or 0
                    ),
                    "current_arm_no_hv_gain_streak_at_decision": int(
                        item.get("current_arm_no_hv_gain_streak_at_decision", 0) or 0
                    ),
                    "chosen_predicted_feasibility": round(
                        safe_float(item.get("chosen_predicted_feasibility", 0.0)),
                        6,
                    ),
                    "chosen_hv_proxy": round(safe_float(item.get("chosen_hv_proxy", 0.0)), 8),
                    "chosen_uncertainty": round(safe_float(item.get("chosen_uncertainty", 0.0)), 8),
                    "chosen_novelty": round(safe_float(item.get("chosen_novelty", 0.0)), 8),
                    "best_feas_arm": str(item.get("best_feas_arm", "")),
                    "best_feas_value": round(safe_float(item.get("best_feas_value", 0.0)), 6),
                    "best_hv_arm": str(item.get("best_hv_arm", "")),
                    "best_hv_value": round(safe_float(item.get("best_hv_value", 0.0)), 8),
                    "best_uncertainty_arm": str(item.get("best_uncertainty_arm", "")),
                    "best_uncertainty_value": round(
                        safe_float(item.get("best_uncertainty_value", 0.0)), 8
                    ),
                    "chosen_vs_best_feas_gap": round(
                        safe_float(item.get("chosen_vs_best_feas_gap", 0.0)),
                        6,
                    ),
                    "chosen_vs_best_hv_gap": round(
                        safe_float(item.get("chosen_vs_best_hv_gap", 0.0)), 8
                    ),
                }
                for item in recent_history[-self.memory_window :]
            ],
        }

    def _empty_llm_fields(self, prompt_version: str) -> Dict[str, Any]:
        return {
            "prompt_version": prompt_version,
            "llm_pipeline_mode": "three_agent",
            "llm_model_requests_this_decision": 0,
            "llm_stage_error": "",
            "llm_feasibility_advocate_json": "",
            "llm_hv_advocate_json": "",
            "llm_arbiter_json": "",
        }

    def _stage_json_fields(self, *, stage_name: str, json_payload: str) -> Dict[str, Any]:
        if stage_name == "feasibility_advocate":
            return {"llm_feasibility_advocate_json": str(json_payload)}
        if stage_name == "hv_advocate":
            return {"llm_hv_advocate_json": str(json_payload)}
        return {"llm_arbiter_json": str(json_payload)}

    def _invoke_stage(
        self,
        *,
        stage_name: str,
        system_prompt: str,
        user_payload: Dict[str, Any],
        acq_key: str,
        text_key: str,
    ) -> Dict[str, Any]:
        try:
            result = self.client.invoke_json(system_prompt, user_payload)
        except (RuntimeError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise _StageInvocationError(stage_name, str(exc)) from exc

        parsed = result.output
        json_payload = compact_json(
            parsed if isinstance(parsed, dict) else {"raw_content": result.raw_content},
            max_chars=1600,
        )
        if not isinstance(parsed, dict):
            raise _StageInvocationError(
                stage_name,
                "output was not a JSON object",
                json_payload=json_payload,
            )
        missing = [key for key in (acq_key, "reason", text_key, "confidence") if key not in parsed]
        if missing:
            raise _StageInvocationError(
                stage_name,
                f"missing keys: {', '.join(missing)}",
                json_payload=json_payload,
            )
        try:
            normalized_acq = _validate_stage_acq(parsed.get(acq_key))
        except ValueError as exc:
            raise _StageInvocationError(stage_name, str(exc), json_payload=json_payload) from exc
        normalized_output = {
            acq_key: normalized_acq,
            "reason": str(parsed.get("reason", "")).strip() or f"{stage_name}_reason_missing",
            text_key: str(parsed.get(text_key, "")).strip() or f"{stage_name}_{text_key}_missing",
            "confidence": min(max(safe_float(parsed.get("confidence", 0.5), 0.5), 0.0), 1.0),
        }
        return {
            "output": normalized_output,
            "json": compact_json(normalized_output, max_chars=1600),
        }

    def _decide_three_agent(self, payload_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt_version = THREE_AGENT_PROMPT_VERSION
        telemetry = self._empty_llm_fields(prompt_version)

        advocate_state_payload = {
            "metadata": payload_state["metadata"],
            "global_state": payload_state["global_state"],
            "per_arm": payload_state["per_arm"],
            "recent_history": payload_state["recent_history"],
        }
        try:
            feasibility_stage = self._invoke_stage(
                stage_name="feasibility_advocate",
                system_prompt=FEASIBILITY_ADVOCATE_PROMPT,
                user_payload={
                    "task": "Recommend the single arm most justified by feasible discovery or feasibility protection right now.",
                    **advocate_state_payload,
                    "output_format": {
                        "recommended_acq": "one of metadata.allowed_arms",
                        "reason": "short snake_case rationale",
                        "argument": "brief feasibility-grounded argument",
                        "confidence": "0..1",
                    },
                    "strict_output_requirement": "JSON object only with keys: recommended_acq, reason, argument, confidence.",
                },
                acq_key="recommended_acq",
                text_key="argument",
            )
            telemetry.update(
                self._stage_json_fields(
                    stage_name="feasibility_advocate",
                    json_payload=str(feasibility_stage["json"]),
                )
            )

            hv_stage = self._invoke_stage(
                stage_name="hv_advocate",
                system_prompt=HV_ADVOCATE_PROMPT,
                user_payload={
                    "task": "Recommend the single arm most justified by feasible hypervolume growth right now.",
                    **advocate_state_payload,
                    "output_format": {
                        "recommended_acq": "one of metadata.allowed_arms",
                        "reason": "short snake_case rationale",
                        "argument": "brief feasible-HV-grounded argument",
                        "confidence": "0..1",
                    },
                    "strict_output_requirement": "JSON object only with keys: recommended_acq, reason, argument, confidence.",
                },
                acq_key="recommended_acq",
                text_key="argument",
            )
            telemetry.update(
                self._stage_json_fields(
                    stage_name="hv_advocate", json_payload=str(hv_stage["json"])
                )
            )

            arbiter_stage = self._invoke_stage(
                stage_name="arbiter",
                system_prompt=ARBITER_PROMPT,
                user_payload={
                    "task": "Choose the single next acquisition arm for the run.",
                    **advocate_state_payload,
                    "feasibility_advocate": feasibility_stage["output"],
                    "hv_advocate": hv_stage["output"],
                    "output_format": {
                        "acq": "one of metadata.allowed_arms",
                        "reason": "short snake_case rationale",
                        "reflection": "brief explanation grounded in the raw state and advocate arguments",
                        "confidence": "0..1",
                    },
                    "strict_output_requirement": "JSON object only with keys: acq, reason, reflection, confidence.",
                },
                acq_key="acq",
                text_key="reflection",
            )
            telemetry.update(
                self._stage_json_fields(
                    stage_name="arbiter", json_payload=str(arbiter_stage["json"])
                )
            )
        except _StageInvocationError as exc:
            raise RuntimeError(f"llm_stage_failed:{exc.stage_name}:{exc}") from exc

        telemetry["llm_model_requests_this_decision"] = 3
        arbiter_output = arbiter_stage["output"]
        return {
            "acq": arbiter_output["acq"],
            "reason": arbiter_output["reason"],
            "reflection": arbiter_output["reflection"],
            "confidence": arbiter_output["confidence"],
            "source": "llm_three_agent",
            "prompt_version": prompt_version,
            **telemetry,
        }

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_provider == "openai" and not self.api_key:
            raise RuntimeError("llm_api_key_missing:OPENAI_API_KEY")
        if not self.client.ready:
            raise RuntimeError(f"llm_client_not_ready:{self.client.init_error or 'unknown'}")

        payload_state = self._build_payload(state)
        return self._decide_three_agent(payload_state)
