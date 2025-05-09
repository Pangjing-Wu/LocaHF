import json
from typing import Dict, Any


def load_json_cfg(config_path) -> Dict[str, Any]:
    if not config_path.exists():
        print(f"[WARN] no config file exist at {config_path}.")
        return {}
    try:
        configs = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        print("[WARN] configs.json is malformed - ignored.")
        return {}
    return configs


def deep_merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict *b* into *a* (b wins)."""
    out = a.copy()
    for k, v in b.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out