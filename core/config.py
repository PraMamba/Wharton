"""Configuration loader with hardcoded fallback defaults."""

import os

_DEFAULTS = {
    "bt": {
        "ot_weight": 0.5,
        "max_iter": 300,
        "tol": 1e-8,
        "regularization": 0.01,
    },
    "rankings": {
        "weights": {
            "bt_strength": 0.40,
            "xgd_per60": 0.30,
            "xg_share": 0.10,
            "gd_pg": 0.10,
            "win_pct": 0.10,
        },
    },
    "cv": {
        "n_folds": 5,
        "random_state": 42,
    },
    "blend": {
        "enet_weight": 0.6,
    },
    "disparity": {
        "min_toi_seconds": 1000,
        "ratio_clip": [0.5, 3.0],
    },
    "prediction": {
        "prob_clip": [0.01, 0.99],
        "bootstrap_integrate": True,
        "uncertainty_shrink_k": 0.0,
    },
    "shrinkage": {
        "K": 500,
    },
    "high_danger": {
        "threshold": 0.20,
    },
    "bt_tuning": {
        "ot_weight_grid": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "regularization_grid": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
        "n_inner_folds": 3,
    },
    "attack_defense": {
        "use_xg": True,
        "regularization": 0.01,
    },
    "rapm": {
        "regularization": 1.0,
        "include_goalies": True,
        "min_toi_unit": 200,
    },
    "augmentation": {
        "symmetric": True,
    },
    "matchup_matrix": {
        "min_toi": 300,
    },
}


def _deep_merge(base, override):
    """Recursively merge override into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path=None):
    """Load config from YAML file, falling back to defaults for missing keys.

    If pyyaml is not installed or the file is missing, returns defaults.
    """
    if path is None:
        # Look for config.yaml in repo root (one level up from core/)
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

    cfg = _DEFAULTS.copy()
    if os.path.exists(path):
        try:
            import yaml
            with open(path, "r") as f:
                user = yaml.safe_load(f) or {}
            cfg = _deep_merge(_DEFAULTS, user)
        except ImportError:
            pass  # pyyaml not installed, use defaults
    return cfg


# Module-level singleton
CFG = load_config()
