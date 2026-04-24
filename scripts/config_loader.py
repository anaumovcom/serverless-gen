"""Load shared configuration (tokens etc.) for download scripts.

Priority for each value (highest wins):
    1. CLI argument (--hf-token / --civitai-token)
    2. Environment variable (HF_TOKEN / HUGGING_FACE_HUB_TOKEN / CIVITAI_TOKEN)
    3. scripts/config.yaml

The config file is intentionally git-ignored so tokens are never committed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPTS_DIR / "config.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict:
    """Parse ``path`` as YAML and return a plain dict. Returns {} on missing file."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "WARNING: pyyaml not installed; config.yaml will not be read.\n"
            "Install with: pip install pyyaml\n"
        )
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_hf_token(cli_value: Optional[str] = None) -> Optional[str]:
    """Return HuggingFace token with priority: CLI → env → config.yaml."""
    if cli_value:
        return cli_value
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    cfg = load_config()
    return cfg.get("hf_token") or None


def resolve_civitai_token(cli_value: Optional[str] = None) -> Optional[str]:
    """Return Civitai token with priority: CLI → env → config.yaml."""
    if cli_value:
        return cli_value
    env = os.environ.get("CIVITAI_TOKEN")
    if env:
        return env
    cfg = load_config()
    return cfg.get("civitai_token") or None
