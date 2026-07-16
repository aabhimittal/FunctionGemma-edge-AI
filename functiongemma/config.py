"""Typed configuration with file + environment overrides.

One config object, three layers of precedence (later wins):
    dataclass defaults  <  YAML/JSON file  <  ``FG_*`` environment variables

Keeping every knob in one typed place (rather than scattered literals) is what
makes a pipeline reproducible: the same config drives data generation, training,
evaluation and serving. YAML is loaded if PyYAML is installed; otherwise a JSON
file with the same keys works, so the library has no hard third-party dependency.
"""

import dataclasses
import json
import os


@dataclasses.dataclass
class Config:
    # --- data / eval ---
    catalog_path: str = "data/tools_catalog.json"
    eval_path: str = "data/eval/golden.jsonl"
    # --- cascade / calibration ---
    tau: float = 0.5
    target_accuracy: float = 0.9
    # --- serving ---
    host: str = "0.0.0.0"
    port: int = 8000
    event_log_path: str = "events.jsonl"
    # --- registry / gate ---
    registry_root: str = "registry"
    min_tool_accuracy: float = 0.8  # promotion gate threshold.


def _load_file(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # optional
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    f"{path} is YAML but PyYAML is not installed; "
                    "install pyyaml or use a .json config"
                ) from exc
            return yaml.safe_load(fh) or {}
        return json.load(fh)


def _coerce(field, value):
    """Coerce a string env value to the field's declared type."""
    if field.type in ("int", int):
        return int(value)
    if field.type in ("float", float):
        return float(value)
    return value


def load_config(path=None):
    """Build a :class:`Config` from defaults, an optional file, then env vars."""
    values = dataclasses.asdict(Config())
    values.update(_load_file(path))

    fields = {f.name: f for f in dataclasses.fields(Config)}
    for name, field in fields.items():
        env = os.environ.get("FG_" + name.upper())
        if env is not None:
            values[name] = _coerce(field, env)

    # Ignore unknown keys so extra file entries don't crash construction.
    return Config(**{k: values[k] for k in fields})
