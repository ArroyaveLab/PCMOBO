"""Constants and policy normalization for the HEA study."""

from __future__ import annotations

from paths import HEA_DATA_PATH, HEA_STUDY_ROOT

HEA_ARM_NAMES = ("pof", "pehvi", "qlognparego", "qucb")
HEA_FIXED_POLICY_NAMES = HEA_ARM_NAMES
HEA_LLM_POLICY_NAMES = ("llm_switch",)
HEA_DYNAMIC_POLICY_NAMES = ("bandit_ucb_switch", *HEA_LLM_POLICY_NAMES)
HEA_POLICY_NAMES = (*HEA_FIXED_POLICY_NAMES, *HEA_DYNAMIC_POLICY_NAMES)

HEA_POLICY_ALIASES = {
    "bandit": "bandit_ucb_switch",
    "bandit_ucb": "bandit_ucb_switch",
    "prob": "pof",
    "ehvi": "pehvi",
    "qlogehvi": "pehvi",
    "pehvi": "pehvi",
    "nparego": "qlognparego",
    "ucb": "qucb",
}

RAW_THRESHOLDS = {
    "density": 9.0,
    "ys": 700.0,
    "pugh": 2.5,
    "st": 2473.0,
}

SCALED_THRESHOLDS = {
    "density": 0.218912147251372,
    "ys": 0.27326687068841815,
    "pugh": 0.34208243243243236,
    "st": 0.3340611001897914,
}

INPUT_COLUMNS = tuple(f"element_{idx:02d}" for idx in range(1, 7))
OBJECTIVE_COLUMNS = (
    "PROP ST (K)",
    "PROP 25C Density (g/cm3)",
    "YS 600 C PRIOR",
    "Pugh_Ratio_PRIOR",
)

POLICY_PROMPT_VERSION = "HEA-Three-Agent"


def normalize_hea_policy_name(name: str) -> str:
    key = str(name).strip().lower()
    return HEA_POLICY_ALIASES.get(key, key)


def is_hea_llm_policy(name: str) -> bool:
    return normalize_hea_policy_name(name) == "llm_switch"


__all__ = [
    "HEA_STUDY_ROOT",
    "HEA_DATA_PATH",
    "HEA_ARM_NAMES",
    "HEA_FIXED_POLICY_NAMES",
    "HEA_LLM_POLICY_NAMES",
    "HEA_DYNAMIC_POLICY_NAMES",
    "HEA_POLICY_NAMES",
    "HEA_POLICY_ALIASES",
    "RAW_THRESHOLDS",
    "SCALED_THRESHOLDS",
    "INPUT_COLUMNS",
    "OBJECTIVE_COLUMNS",
    "POLICY_PROMPT_VERSION",
    "normalize_hea_policy_name",
    "is_hea_llm_policy",
]
