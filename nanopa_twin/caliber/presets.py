from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from nanopa_twin.caliber.specs import ExperimentConfig

T = TypeVar("T")

PRESET_DIR = Path(__file__).resolve().parent / "presets"


def _coerce(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is tuple:
        inner = get_args(annotation)
        if len(inner) == 2 and inner[1] is Ellipsis:
            return tuple(_coerce(item, inner[0]) for item in value)
        return tuple(_coerce(item, ann) for item, ann in zip(value, inner, strict=True))
    if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        return _build(annotation, value)
    if annotation is float and isinstance(value, (int, float)):
        return float(value)
    return value


def _build(cls: type[T], payload: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    unknown = set(payload) - set(hints)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, raw in payload.items():
        kwargs[name] = _coerce(raw, hints[name])
    return cls(**kwargs)


def from_mapping(payload: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, payload)


def resolve_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    named = PRESET_DIR / f"{name_or_path}.yaml"
    if named.exists():
        return named
    raise FileNotFoundError(f"no preset named {name_or_path!r} in {PRESET_DIR}")


def load_preset(name_or_path: str) -> ExperimentConfig:
    path = resolve_path(name_or_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return from_mapping(payload)


def to_mapping(config: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def available_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))
