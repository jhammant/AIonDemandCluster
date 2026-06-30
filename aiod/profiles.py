"""Named spin-up presets ("profiles").

A profile bundles everything needed to stand up a particular stack — model,
quant, provider, price/GPU prefs, idle timeout, vLLM tweaks — under one name, so
you can `aiod spin --profile coder` instead of repeating flags.

Resolution order (later wins): built-in presets  <  user file  <  CLI flags.
The user file is YAML at ~/.config/aiod/profiles.yaml and is meant to be
hand-edited — adding a new architecture is just a new block in that file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml
from platformdirs import user_config_dir

PROFILE_FILE = Path(user_config_dir("aiod", appauthor=False)) / "profiles.yaml"


@dataclass
class Profile:
    name: str
    model: str
    provider: str = "vast"  # vast | runpod | ...
    quant: str = "bf16"  # bf16 | fp8 | awq-int4 | gptq-int4
    max_price: float | None = None  # $/hr cap; None -> use global default
    num_gpus: int | None = None  # override auto-sizing's GPU count
    context: int | None = None  # max model length to serve
    concurrency: int = 4  # concurrent seqs assumed for VRAM sizing
    ttl_hours: float | None = None  # auto-destroy reminder; None -> global default
    idle_minutes: int | None = None  # auto-shutdown after this idle window
    tool_call_parser: str | None = None  # vLLM --tool-call-parser; None = use model_configs
    extra_vllm_args: list[str] = field(default_factory=list)
    optimizations: list[str] = field(default_factory=list)  # opt selection tokens (KEY or KEY=VAL)
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> Profile:
        known = {f.name for f in fields(cls)} - {"name"}
        return cls(name=name, **{k: v for k, v in d.items() if k in known})

    def body(self) -> dict:
        d = asdict(self)
        d.pop("name")
        return d


# Built-in presets. Models chosen for strong tool-calling (Claude Code is very
# tool-heavy). Tweak/extend via the user YAML file.
BUILTIN: dict[str, Profile] = {
    p.name: p
    for p in [
        Profile(
            name="coder-7b",
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            quant="bf16",
            max_price=0.6,
            idle_minutes=20,
            description="Cheapest end-to-end test — fits one 24GB GPU.",
        ),
        Profile(
            name="coder-32b",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            quant="fp8",
            idle_minutes=20,
            description="Strong coding + tool calling; one 80GB GPU at fp8.",
        ),
        Profile(
            name="qwen3-coder-30b",
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            quant="fp8",
            idle_minutes=20,
            description="Fast MoE, strong agentic behavior.",
        ),
        Profile(
            name="glm-4.6",
            model="zai-org/GLM-4.6",
            quant="fp8",
            idle_minutes=20,
            description="Capable agentic/coding model (large).",
        ),
    ]
}


def _load_user() -> dict[str, Profile]:
    if not PROFILE_FILE.exists():
        return {}
    try:
        data = yaml.safe_load(PROFILE_FILE.read_text()) or {}
    except yaml.YAMLError:
        return {}
    profiles: dict[str, Profile] = {}
    for name, body in (data.get("profiles") or {}).items():
        if isinstance(body, dict) and body.get("model"):
            profiles[name] = Profile.from_dict(name, body)
    return profiles


def all_profiles() -> dict[str, Profile]:
    """Built-ins overlaid with the user file (user wins on name clash)."""
    merged = dict(BUILTIN)
    merged.update(_load_user())
    return merged


def get(name: str) -> Profile | None:
    return all_profiles().get(name)


def is_builtin(name: str) -> bool:
    return name in BUILTIN and name not in _load_user()


def save(profile: Profile) -> None:
    """Write/overwrite a profile in the user YAML file."""
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if PROFILE_FILE.exists():
        data = yaml.safe_load(PROFILE_FILE.read_text()) or {}
    data.setdefault("profiles", {})
    data["profiles"][profile.name] = profile.body()
    PROFILE_FILE.write_text(yaml.safe_dump(data, sort_keys=False))


def remove(name: str) -> bool:
    """Delete a user profile. Returns False if it isn't in the user file."""
    if not PROFILE_FILE.exists():
        return False
    data = yaml.safe_load(PROFILE_FILE.read_text()) or {}
    profs = data.get("profiles") or {}
    if name not in profs:
        return False
    del profs[name]
    data["profiles"] = profs
    PROFILE_FILE.write_text(yaml.safe_dump(data, sort_keys=False))
    return True
