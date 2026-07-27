import functools
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"


@functools.lru_cache
def _load_yaml(name: str) -> dict:
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt file for '{name}' at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render_prompt(name: str, key: str = "system_prompt", **kwargs) -> str:
    """Render prompts/{name}.yaml[key] with the given placeholders.

    Raises a clear error (instead of a bare KeyError) when the key is
    missing from the file, or a placeholder used in the template is
    missing from kwargs.
    """
    prompts = _load_yaml(name)
    if key not in prompts:
        raise KeyError(f"Prompt '{name}' has no key '{key}' (has: {list(prompts)})")
    template = prompts[key]
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise KeyError(
            f"Missing placeholder {exc} when rendering '{name}.{key}'"
        ) from exc
