"""Switch registry, exact parameter counts and ablation labels per model family."""
from __future__ import annotations

import math
from typing import Iterable, Mapping

from .schemas import ConfigDiff, ConfigValue, Switch
from .store import LabError

INPUT_DIMS = {"RAW": 768, "GEO": 42, "HYBRID": 810}
# GEO = 2 named columns per cvslab.facts.GEOMETRY_FAMILIES entry (value rescaled by the family's
# own bucket-3 threshold, bucket/3, sign-flipped for black to move); HYBRID = RAW ++ GEO.

# Config keys that are not experimental switches but must still travel with a
# configuration and be covered by its hash (e.g. the frozen supervision spec).
RESERVED_CONFIG_KEYS = frozenset({"TARGET_SPEC"})

NNUE_SWITCHES: list[Switch] = [
    Switch(
        key="INPUT", label="Input representation", kind="enum", default="RAW",
        choices=["RAW", "GEO", "HYBRID"], axis="representation",
        description="RAW = 768 piece-square planes from the side to move (mirror + colour swap for "
        "black). GEO = the 42 named deterministic geometry columns of the cvslab.facts registry "
        "(White-POV delta / the family's own bucket-3 threshold, bucket/3, sign-flipped for black). "
        "HYBRID = RAW ++ GEO in one input vector. All three are side-to-move relative.",
    ),
    Switch(
        key="ARCH", label="Architecture", kind="enum", default="crelu1", choices=["crelu1", "linear"],
        axis="capacity",
        description="crelu1 = inputs -> H clipped-ReLU -> 1 (learned params = inputs*H + 2H + 1). "
        "linear = inputs -> 1 directly, the white-box semantic floor (learned params = inputs + 1; "
        "H is ignored).",
    ),
    Switch(
        key="H", label="Hidden width", kind="int", default=1, min=1, max=4096, axis="capacity",
        description="Neurons in the single clipped-ReLU hidden layer. Learned parameters = inputs*H + 2H + 1.",
    ),
    Switch(
        key="OUT_SCALE_CP", label="Output scale (cp)", kind="float", default=400.0, min=1.0, max=10000.0, axis="capacity",
        description="Fixed, unlearned multiplier from network output to centipawns.",
    ),
    Switch(key="EPOCHS", label="Epochs", kind="int", default=40, min=1, max=10000, axis="compute",
           description="Full passes over the frozen training split."),
    Switch(key="BATCH", label="Batch size", kind="int", default=256, min=1, max=1_000_000, axis="training"),
    # S8 student-compute control: 0 keeps the historical epoch-bounded schedule; a positive
    # value runs EXACTLY that many optimizer updates, so arms with different dataset sizes
    # receive identical student-side optimization work.
    Switch(key="MAX_UPDATES", label="Max optimizer updates", kind="int", default=0, min=0,
           max=10_000_000, axis="compute",
           description="0 = epoch-bounded (historical); >0 = exactly this many optimizer updates"),
    Switch(key="LR", label="Learning rate", kind="float", default=0.003, min=1e-7, max=1.0, axis="training"),
    Switch(key="OPTIMIZER", label="Optimizer", kind="enum", default="adam", choices=["adam", "sgd"], axis="training"),
    Switch(key="INIT_STD", label="Input weight init std", kind="float", default=0.05, min=0.0, max=1.0, axis="training"),
    Switch(
        key="LAMBDA", label="Eval / result blend", kind="float", default=1.0, min=0.0, max=1.0, axis="signal",
        description="Target = LAMBDA*sigmoid(cp/K) + (1-LAMBDA)*result. Rows without a game result use the eval target only.",
    ),
    Switch(key="K", label="Sigmoid scale", kind="float", default=256.0, min=1.0, max=10000.0, axis="signal",
           description="Centipawn scale of the win-probability sigmoid used by the training target."),
]

FAMILIES: dict[str, list[Switch]] = {"NNUE": NNUE_SWITCHES}


def switches(family: str) -> list[Switch]:
    try:
        return FAMILIES[family]
    except KeyError:
        raise LabError(f"unknown family {family!r}; registered families: {', '.join(FAMILIES)}") from None


def switch_map(family: str) -> dict[str, Switch]:
    return {sw.key: sw for sw in switches(family)}


def coerce(sw: Switch, value: object) -> ConfigValue:
    try:
        if sw.kind == "int":
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            result: ConfigValue = int(value.strip()) if isinstance(value, str) else int(value)
        elif sw.kind == "float":
            if isinstance(value, bool):
                raise ValueError
            result = float(value)
            if not math.isfinite(result):
                raise ValueError
        elif sw.kind == "bool":
            text = str(value).strip().lower()
            if isinstance(value, bool):
                result = value
            elif text in {"1", "true", "on", "yes"}:
                result = True
            elif text in {"0", "false", "off", "no"}:
                result = False
            else:
                raise ValueError
        else:
            result = str(value)
            if result not in sw.choices:
                raise LabError(f"{sw.key}={result!r} is not one of {sw.choices}")
    except (TypeError, ValueError):
        raise LabError(f"{sw.key}: cannot interpret {value!r} as {sw.kind}") from None
    if sw.kind in ("int", "float"):
        if sw.min is not None and result < sw.min:
            raise LabError(f"{sw.key}={result} is below the registered minimum {sw.min:g}")
        if sw.max is not None and result > sw.max:
            raise LabError(f"{sw.key}={result} is above the registered maximum {sw.max:g}")
    return result


def check_registered(family: str, keys: Iterable[str]) -> None:
    registry = switch_map(family)
    unknown = sorted(k for k in keys if k not in registry and k not in RESERVED_CONFIG_KEYS)
    if unknown:
        raise LabError(f"unregistered switch(es) for {family}: {', '.join(unknown)}; register a switch before using it")


def merge_config(family: str, base: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, ConfigValue]:
    """Full effective configuration: every registered switch, hidden defaults made explicit."""
    check_registered(family, [*base, *overrides])
    merged: dict[str, ConfigValue] = {}
    for key in RESERVED_CONFIG_KEYS:  # metadata travels with the config and is hashed
        raw = overrides.get(key, base.get(key))
        if raw is not None:
            merged[key] = raw
    for sw in switches(family):
        raw = overrides[sw.key] if sw.key in overrides else base.get(sw.key, sw.default)
        merged[sw.key] = coerce(sw, raw)
    return merged


def diff_configs(family: str, control: Mapping[str, ConfigValue], config: Mapping[str, ConfigValue]) -> list[ConfigDiff]:
    return [
        ConfigDiff(key=sw.key, control=control.get(sw.key), value=config[sw.key], axis=sw.axis)
        for sw in switches(family)
        if control.get(sw.key) != config.get(sw.key)
    ]


def parameter_shapes(family: str, config: Mapping[str, ConfigValue]) -> dict[str, list[int]]:
    if family != "NNUE":
        raise LabError(f"no parameter model registered for family {family}")
    inputs, hidden = INPUT_DIMS[str(config["INPUT"])], int(config["H"])
    if str(config.get("ARCH", "crelu1")) == "linear":
        return {"w": [inputs], "b": []}
    return {"w1": [inputs, hidden], "b1": [hidden], "w2": [hidden], "b2": []}


def count_parameters(shapes: Mapping[str, list[int]]) -> int:
    return sum(math.prod(shape) for shape in shapes.values())


def param_count(family: str, config: Mapping[str, ConfigValue]) -> int:
    return count_parameters(parameter_shapes(family, config))


def format_params(n: int) -> str:
    if n < 1000:
        return str(n)
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= divisor:
            value = n / divisor
            text = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            return text.removesuffix(".0") + suffix
    raise AssertionError("unreachable")


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def display_label(obj_id: str, family: str, generation: str, n_params: int, baseline_name: str,
                  overrides: Iterable[tuple[str, object]]) -> str:
    """`A#### FAMILY-GEN-PARAMS @ BASELINE : OVERRIDES` — only deviations from the baseline."""
    label = f"{obj_id} {family}-{generation}-P{format_params(n_params)} @ {baseline_name}"
    parts = [f"{key}={format_value(value)}" for key, value in overrides]
    return f"{label} : {','.join(parts)}" if parts else label
