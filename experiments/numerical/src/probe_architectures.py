from __future__ import annotations

LEGACY_CIRCULAR_PROBE_ARCHITECTURE = "circular"
CIRCULAR_REG_PROBE_ARCHITECTURE = "circular_reg"
CIRCULAR_CLS_PROBE_ARCHITECTURE = "circular_cls"

_DISPLAY_NAMES = {
    "linear": "Linear probe",
    "pls": "PLS probe",
    "pls_x_weight1": "PLS x_weight1 probe",
    CIRCULAR_REG_PROBE_ARCHITECTURE: "Circular probe (regression)",
    CIRCULAR_CLS_PROBE_ARCHITECTURE: "Circular probe (classification)",
    "digitplace_linprobe_cat": "Linear probe (categorical)",
    "digitplace_linprobe_reg": "Linear probe (regression)",
}


def canonicalize_probe_architecture_name(architecture: str) -> str:
    name = str(architecture)
    if name == LEGACY_CIRCULAR_PROBE_ARCHITECTURE:
        return CIRCULAR_REG_PROBE_ARCHITECTURE
    return name


def canonicalize_probe_identifier(name: str) -> str:
    value = str(name)
    if value.startswith("random_norm_match__"):
        prefix, architecture = value.split("__", 1)
        return f"{prefix}__{canonicalize_probe_architecture_name(architecture)}"
    if value.startswith("probe_"):
        prefix, architecture = value.split("_", 1)
        return f"{prefix}_{canonicalize_probe_architecture_name(architecture)}"
    return canonicalize_probe_architecture_name(value)


def probe_architecture_display_name(architecture: str) -> str:
    canonical = canonicalize_probe_architecture_name(architecture)
    return _DISPLAY_NAMES.get(canonical, canonical)
