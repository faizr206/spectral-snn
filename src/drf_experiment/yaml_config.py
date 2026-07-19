from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {path}")
    return payload


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    current_list_key: str | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                if current_list_key is None:
                    raise ValueError(f"List item without a key in {path}: {raw_line.rstrip()}")
                payload[current_list_key].append(_parse_scalar(stripped[2:]))
                continue
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line in {path}: {raw_line.rstrip()}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                payload[key] = []
                current_list_key = key
            else:
                payload[key] = _parse_scalar(value)
                current_list_key = None
    return payload


def list_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Expected a list or comma-separated string, got {type(value).__name__}")
