"""LLM switch policy for the HEA finite-pool portfolio."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from core import compact_json, safe_float
from llm_runtime import LangGraphJSONClient
from studies.hea_design.constants import HEA_ARM_NAMES, POLICY_PROMPT_VERSION

ALLOWED_ACQS = HEA_ARM_NAMES
POLICY_NAME = "llm_switch"
POLICY_INTENT = (
    "Primary goal: maximize feasible fixed-range hypervolume over the run while protecting feasible "
    "discovery when evidence is weak. Infer strategic tradeoffs from the current evidence and recent "
    "history."
)
COMMON_PROMPT_PREAMBLE = (
    "You are part of a three-agent portfolio switch for constrained materials design. "
    "Choose exactly one arm from: pof, pehvi, qlognparego, qucb. "
)
COMMON_ARM_SEMANTICS = (
    "All arms operate in a BCC-aware usable-value space. "
    "Arm semantics: pof=objective-feasibility base score weighted by p_bcc; "
    "pehvi=pool-scored pEHVI weighted by p_bcc; "
    "qlognparego=pool-scored qLogNParEGO weighted by p_bcc; "
    "qucb=pool-scored q-UCB weighted by p_bcc. "
)
COMMON_JSON_REQUIREMENT = (
    "Use only metadata, global_state, per_arm, and recent_history. Return strict JSON only."
)


def build_policy_prompts() -> Dict[str, str]:
    visibility_block = (
        "Explicit budget information is intentionally hidden; infer timing only from the non-budget "
        "state that is provided. "
    )
    intent_block = f"Policy intent: {POLICY_INTENT} "
    feasibility_suffix = (
        "Your role is to judge which arm best protects feasible discovery and BCC-safe progress right now. "
        "Do not assume any fixed mapping from early or late stages to a specific arm; infer the best arm from the evidence. "
        "Return JSON with keys: recommended_acq, reason, argument, confidence."
    )
    hv_suffix = (
        "Your role is to judge which arm best supports fixed-range feasible hypervolume growth right now. "
        "Do not assume any fixed mapping from early or late stages to a specific arm; infer the best arm from the evidence. "
        "Return JSON with keys: recommended_acq, reason, argument, confidence."
    )
    arbiter_suffix = (
        "Your role is to choose the single arm that best serves the run-level objective over time. "
        "Use the advocate outputs as evidence, not commands. Do not hard-code stage-to-arm rules; resolve the tradeoff from the raw state and the active policy intent. "
        "Return JSON with keys: acq, reason, reflection, confidence."
    )
    prefix = COMMON_PROMPT_PREAMBLE + COMMON_ARM_SEMANTICS + intent_block + visibility_block
    return {
        "feasibility_advocate": prefix + feasibility_suffix + " " + COMMON_JSON_REQUIREMENT,
        "hv_advocate": prefix + hv_suffix + " " + COMMON_JSON_REQUIREMENT,
        "arbiter": prefix + arbiter_suffix + " " + COMMON_JSON_REQUIREMENT,
    }


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
        key, value = line.split("=", 1)
        k = key.strip()
        v = value.strip().strip("\"'")
        if k and k not in os.environ:
            os.environ[k] = v


def normalize_mode(value: str) -> str:
    raw = str(value).strip().lower()
    if raw in ALLOWED_ACQS:
        return raw
    return {
        "prob": "pof",
        "ehvi": "pehvi",
        "qlogehvi": "pehvi",
        "pehvi": "pehvi",
        "nparego": "qlognparego",
        "ucb": "qucb",
    }.get(raw, raw)


def _validate_stage_acq(value: Any) -> str:
    out = normalize_mode(str(value))
    if out not in ALLOWED_ACQS:
        raise ValueError(f"invalid acquisition '{value}'")
    return out


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
        self.prompts = build_policy_prompts()
        self.prompt_version = f"{POLICY_PROMPT_VERSION}:{POLICY_NAME}"
        self.client = LangGraphJSONClient(
            model=self.model,
            provider=self.llm_provider,
            env_path=self.env_path,
            timeout_s=self.timeout_s,
            temperature=self.temperature,
        )

    def _history_item(
        self,
        item: Dict[str, Any],
        *,
        steps_ago: int,
        budget_visible: bool,
    ) -> Dict[str, Any]:
        payload = {
            "chosen_acq": str(item.get("chosen_acq", "")),
            "observed_feasible": int(item.get("observed_feasible", 0) or 0),
            "full_measurement_observed": int(item.get("full_measurement_observed", 0) or 0),
            "feasible_gain": int(item.get("feasible_gain", 0) or 0),
            "hv_scaled_gain": round(safe_float(item.get("hv_scaled_gain", 0.0)), 8),
            "cumulative_feasible_count": int(item.get("cumulative_feasible_count", 0) or 0),
            "chosen_pred_total_feas_prob": round(
                safe_float(item.get("chosen_pred_total_feas_prob", 0.0)),
                6,
            ),
            "chosen_pred_bcc_prob": round(safe_float(item.get("chosen_pred_bcc_prob", 0.0)), 6),
            "chosen_pred_obj_score": round(safe_float(item.get("chosen_pred_obj_score", 0.0)), 8),
            "chosen_base_acq_value": round(safe_float(item.get("chosen_base_acq_value", 0.0)), 8),
            "chosen_acq_value": round(safe_float(item.get("chosen_acq_value", 0.0)), 8),
            "best_feas_arm": str(item.get("best_feas_arm", "")),
            "best_feas_value": round(safe_float(item.get("best_feas_value", 0.0)), 6),
            "best_hv_arm": str(item.get("best_hv_arm", "")),
            "best_hv_value": round(safe_float(item.get("best_hv_value", 0.0)), 8),
        }
        if budget_visible:
            payload["iteration"] = int(item.get("iteration", 0))
        else:
            payload["steps_ago"] = int(steps_ago)
        return payload

    def _build_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metadata = state.get("metadata", {}) if isinstance(state.get("metadata"), dict) else {}
        global_state = (
            state.get("global_state", {}) if isinstance(state.get("global_state"), dict) else {}
        )
        per_arm = state.get("per_arm", {}) if isinstance(state.get("per_arm"), dict) else {}
        recent_history = (
            state.get("recent_history", []) if isinstance(state.get("recent_history"), list) else []
        )
        budget_visible = False
        payload = {
            "metadata": {
                "benchmark": str(metadata.get("benchmark", "hea_design")),
                "allowed_arms": [str(x) for x in metadata.get("allowed_arms", list(ALLOWED_ACQS))],
            },
            "global_state": {
                "cumulative_feasible_count": int(
                    global_state.get("cumulative_feasible_count", 0) or 0
                ),
                "cumulative_measured": int(global_state.get("cumulative_measured", 0) or 0),
                "cumulative_phase_observed": int(
                    global_state.get("cumulative_phase_observed", 0) or 0
                ),
                "current_hv_scaled_fixed_range": round(
                    safe_float(global_state.get("current_hv_scaled_fixed_range", 0.0)),
                    8,
                ),
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
                    "base_acq_value": round(safe_float(values.get("base_acq_value", 0.0)), 8),
                    "pred_total_feas_prob": round(
                        safe_float(values.get("pred_total_feas_prob", 0.0)), 6
                    ),
                    "pred_obj_feas_prob": round(
                        safe_float(values.get("pred_obj_feas_prob", 0.0)), 6
                    ),
                    "pred_bcc_prob": round(safe_float(values.get("pred_bcc_prob", 0.0)), 6),
                    "pred_obj_score": round(safe_float(values.get("pred_obj_score", 0.0)), 8),
                    "normalized_acq_value": round(
                        safe_float(values.get("normalized_acq_value", 0.0)), 6
                    ),
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
                }
                for arm, values in per_arm.items()
                if arm in ALLOWED_ACQS
            },
            "recent_history": [],
        }
        window = recent_history[-self.memory_window :]
        payload["recent_history"] = [
            self._history_item(
                item,
                steps_ago=(len(window) - 1 - idx),
                budget_visible=budget_visible,
            )
            for idx, item in enumerate(window)
        ]
        return payload

    def build_audit_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._build_payload(state)

    def _policy_fields(self) -> Dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "llm_policy_name": POLICY_NAME,
        }

    def _empty_llm_fields(self) -> Dict[str, Any]:
        return {
            **self._policy_fields(),
            "llm_pipeline_mode": "three_agent",
            "llm_payload_transmitted": 0,
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
        payload: Dict[str, Any],
        acq_key: str,
        text_key: str,
    ) -> Dict[str, Any]:
        try:
            result = self.client.invoke_json(system_prompt, payload)
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
        output = {
            acq_key: normalized_acq,
            "reason": str(parsed.get("reason", "")).strip() or f"{stage_name}_reason_missing",
            text_key: str(parsed.get(text_key, "")).strip() or f"{stage_name}_{text_key}_missing",
            "confidence": min(max(safe_float(parsed.get("confidence", 0.5), 0.5), 0.0), 1.0),
        }
        return {"output": output, "json": compact_json(output, max_chars=1600)}

    def _decide_three_agent(self, payload_state: Dict[str, Any]) -> Dict[str, Any]:
        telemetry = self._empty_llm_fields()
        advocate_payload = {
            "metadata": payload_state["metadata"],
            "global_state": payload_state["global_state"],
            "per_arm": payload_state["per_arm"],
            "recent_history": payload_state["recent_history"],
        }
        try:
            feasibility = self._invoke_stage(
                stage_name="feasibility_advocate",
                system_prompt=self.prompts["feasibility_advocate"],
                payload={
                    "task": "Recommend the single arm most justified by feasible discovery and BCC-safe progress right now.",
                    **advocate_payload,
                    "output_format": {
                        "recommended_acq": "one of metadata.allowed_arms",
                        "reason": "short snake_case rationale",
                        "argument": "brief feasibility-grounded argument",
                        "confidence": "0..1",
                    },
                },
                acq_key="recommended_acq",
                text_key="argument",
            )
            telemetry.update(
                self._stage_json_fields(
                    stage_name="feasibility_advocate",
                    json_payload=str(feasibility["json"]),
                )
            )
            hv = self._invoke_stage(
                stage_name="hv_advocate",
                system_prompt=self.prompts["hv_advocate"],
                payload={
                    "task": "Recommend the single arm most justified by fixed-range feasible hypervolume growth right now.",
                    **advocate_payload,
                    "output_format": {
                        "recommended_acq": "one of metadata.allowed_arms",
                        "reason": "short snake_case rationale",
                        "argument": "brief HV-grounded argument",
                        "confidence": "0..1",
                    },
                },
                acq_key="recommended_acq",
                text_key="argument",
            )
            telemetry.update(
                self._stage_json_fields(stage_name="hv_advocate", json_payload=str(hv["json"]))
            )
            arbiter = self._invoke_stage(
                stage_name="arbiter",
                system_prompt=self.prompts["arbiter"],
                payload={
                    "task": "Choose the next arm for the HEA run.",
                    **advocate_payload,
                    "feasibility_advocate": feasibility["output"],
                    "hv_advocate": hv["output"],
                    "output_format": {
                        "acq": "one of metadata.allowed_arms",
                        "reason": "short snake_case rationale",
                        "reflection": "brief explanation grounded in the raw state and advocate arguments",
                        "confidence": "0..1",
                    },
                },
                acq_key="acq",
                text_key="reflection",
            )
            telemetry.update(
                self._stage_json_fields(stage_name="arbiter", json_payload=str(arbiter["json"]))
            )
        except _StageInvocationError as exc:
            raise RuntimeError(f"llm_stage_failed:{exc.stage_name}:{exc}") from exc

        telemetry.update({"llm_payload_transmitted": 1, "llm_model_requests_this_decision": 3})
        out = arbiter["output"]
        return {
            "acq": out["acq"],
            "reason": out["reason"],
            "reflection": out["reflection"],
            "confidence": out["confidence"],
            "source": "llm_three_agent",
            **telemetry,
        }

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_provider == "openai" and not self.api_key:
            raise RuntimeError("llm_api_key_missing:OPENAI_API_KEY")
        if not self.client.ready:
            raise RuntimeError(f"llm_client_not_ready:{self.client.init_error or 'unknown'}")
        payload = self._build_payload(state)
        return self._decide_three_agent(payload)
