"""Constants and policy normalization for the ESTM study."""

from __future__ import annotations

from paths import ESTM_DATA_PATH, ESTM_MANIFEST_PATH, ESTM_STUDY_ROOT

ESTM_ARM_NAMES = ("qlogpof", "qlogehvi", "qlognparego", "qucb")
ESTM_FIXED_POLICY_NAMES = ESTM_ARM_NAMES
ESTM_LLM_POLICY_NAMES = ("llm_switch",)
ESTM_DYNAMIC_POLICY_NAMES = ("bandit_ucb_switch", *ESTM_LLM_POLICY_NAMES)
ESTM_POLICY_NAMES = (*ESTM_FIXED_POLICY_NAMES, *ESTM_DYNAMIC_POLICY_NAMES)

ESTM_POLICY_ALIASES = {
    "bandit": "bandit_ucb_switch",
    "bandit_ucb": "bandit_ucb_switch",
    "pof": "qlogpof",
    "prob": "qlogpof",
    "ehvi": "qlogehvi",
    "pehvi": "qlogehvi",
    "qehvi": "qlogehvi",
    "nparego": "qlognparego",
    "ucb": "qucb",
}

FEATURE_COLUMNS = (
    "feat_mean_atomic_number",
    "feat_mean_atomic_volume",
    "feat_mean_atomic_weight",
    "feat_std_atomic_number",
    "feat_std_atomic_volume",
    "feat_std_atomic_weight",
    "feat_max_atomic_number",
    "feat_max_atomic_volume",
    "feat_max_atomic_weight",
    "feat_min_atomic_number",
    "feat_min_atomic_volume",
    "feat_min_atomic_weight",
    "feat_temperature_norm",
)

OBJECTIVE_COLUMNS = (
    "obj_absS_scaled",
    "obj_log10_sigma_scaled",
    "obj_neg_kappa_scaled",
)

CONSTRAINT_COLUMNS = (
    "con_zt_slack",
    "con_pf_slack",
)

RAW_PROPERTY_COLUMNS = (
    "absS_uV_per_K",
    "electrical_conductivity_S_per_m",
    "log10_sigma",
    "thermal_conductivity_W_per_mK",
    "power_factor_W_per_mK2",
    "ZT",
)

ESTM_TASK_DEFINITION = {
    "dataset_version": "v1",
    "temperature_band": "mid",
    "temperature_min_k": 300.0,
    "temperature_max_k": 600.0,
    "zt_threshold": 0.8,
    "power_factor_threshold": 1.2e-3,
}

POLICY_PROMPT_VERSION = "ESTM-Three-Agent-v1"


def normalize_estm_policy_name(name: str) -> str:
    key = str(name).strip().lower()
    return ESTM_POLICY_ALIASES.get(key, key)


def is_estm_llm_policy(name: str) -> bool:
    return normalize_estm_policy_name(name) == "llm_switch"


__all__ = [
    "ESTM_STUDY_ROOT",
    "ESTM_DATA_PATH",
    "ESTM_MANIFEST_PATH",
    "ESTM_ARM_NAMES",
    "ESTM_FIXED_POLICY_NAMES",
    "ESTM_LLM_POLICY_NAMES",
    "ESTM_DYNAMIC_POLICY_NAMES",
    "ESTM_POLICY_NAMES",
    "ESTM_POLICY_ALIASES",
    "FEATURE_COLUMNS",
    "OBJECTIVE_COLUMNS",
    "CONSTRAINT_COLUMNS",
    "RAW_PROPERTY_COLUMNS",
    "ESTM_TASK_DEFINITION",
    "POLICY_PROMPT_VERSION",
    "normalize_estm_policy_name",
    "is_estm_llm_policy",
]
