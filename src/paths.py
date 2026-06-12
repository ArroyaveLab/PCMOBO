"""Filesystem paths for PCMOBO."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
STUDIES_ROOT = SOURCE_ROOT / "studies"

HEA_STUDY_ROOT = STUDIES_ROOT / "hea_design"
HEA_DATA_PATH = HEA_STUDY_ROOT / "data" / "design_space.csv"

ESTM_STUDY_ROOT = STUDIES_ROOT / "estm_thermoelectric"
ESTM_DATA_PATH = ESTM_STUDY_ROOT / "data" / "estm_mid_constrained_mobo_v1.csv"
ESTM_MANIFEST_PATH = ESTM_STUDY_ROOT / "data" / "estm_mid_constrained_mobo_v1_manifest.json"

RUNS_ROOT = (
    Path(os.environ["PCMBO_OUTPUT_ROOT"])
    if "PCMBO_OUTPUT_ROOT" in os.environ
    else Path.cwd() / "runs"
)
SYNTHETIC_RUNS_ROOT = RUNS_ROOT / "synthetic"
HEA_RUNS_ROOT = RUNS_ROOT / "hea_design"
ESTM_RUNS_ROOT = RUNS_ROOT / "estm_thermoelectric"
