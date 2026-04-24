#!/usr/bin/env python3
"""Download all required models defined in models.yaml.

Usage
-----

    # Download everything
    python scripts/download_all_models.py

    # Only models for one workflow
    python scripts/download_all_models.py --workflow flux2_reactor
    python scripts/download_all_models.py --workflow wan22_i2v

    # Create directory structure only (no downloads)
    python scripts/download_all_models.py --init-dirs

    # Override volume root (default from models.yaml or RUNPOD_VOLUME_DIR env)
    python scripts/download_all_models.py --volume-root /mnt/my-volume

    # Force re-download even if file exists
    python scripts/download_all_models.py --force

    # Skip specific model IDs
    python scripts/download_all_models.py --skip inswapper_128 --skip flux2_text_encoder

    # Dry run (show what would be downloaded, no actual requests)
    python scripts/download_all_models.py --dry-run

Dependencies
------------

    pip install -r scripts/requirements.txt
    # (requests, tqdm, pyyaml)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import yaml  # type: ignore
except ImportError as exc:
    sys.stderr.write("ERROR: 'pyyaml' is required. Install with: pip install pyyaml\n")
    raise SystemExit(1) from exc

try:
    from download_model import download
except ImportError as exc:
    sys.stderr.write(
        f"ERROR: could not import download_model.py ({exc}). "
        "Make sure you run this script from the repo root or the scripts/ directory.\n"
    )
    raise SystemExit(1) from exc

from config_loader import resolve_civitai_token, resolve_hf_token


MANIFEST = SCRIPTS_DIR / "models.yaml"

COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_CYAN = "\033[36m"
COLOR_BOLD = "\033[1m"


def _colored(text: str, color: str) -> str:
    if not sys.stderr.isatty():
        return text
    return f"{color}{text}{COLOR_RESET}"


def _load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping at the top level")
    return data


def _resolve_volume_root(manifest: dict, cli_override: Optional[str]) -> Path:
    if cli_override:
        return Path(cli_override)
    env = os.environ.get("RUNPOD_VOLUME_DIR")
    if env:
        return Path(env)
    return Path(manifest.get("volume_root", "/runpod-volume"))


def _create_dirs(volume_root: Path, dirs: list[str], *, dry_run: bool) -> None:
    print(_colored(f"\n{'[DRY-RUN] ' if dry_run else ''}Creating directory structure under {volume_root}/", COLOR_BOLD))
    for rel in dirs:
        full = volume_root / rel
        if dry_run:
            print(f"  [mkdir] {full}")
        else:
            full.mkdir(parents=True, exist_ok=True)
            print(f"  {_colored('✓', COLOR_GREEN)} {full}")


def _filter_models(
    models: list[dict],
    workflow: Optional[str],
    skip_ids: set[str],
) -> list[dict]:
    result = []
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id", "")
        if mid in skip_ids:
            continue
        if workflow is not None:
            workflows = m.get("workflows") or []
            if workflow not in workflows:
                continue
        result.append(m)
    return result


def _print_summary_line(model: dict, volume_root: Path) -> None:
    dest = volume_root / str(model.get("dest", ""))
    filename = model.get("filename", "")
    full_path = dest / filename
    url = model.get("url") or ""
    label = model.get("id", "?")
    workflows = ", ".join(model.get("workflows") or [])
    print(
        f"  {_colored(label, COLOR_CYAN):<30}  {workflows:<20}  "
        f"{_colored(filename, COLOR_BOLD)}"
    )
    print(f"    dest  : {full_path}")
    if url:
        print(f"    url   : {url}")
    else:
        print(f"    url   : {_colored('NOT SET — skip', COLOR_YELLOW)}")
    notes = model.get("notes")
    if notes:
        note_str = " ".join(str(notes).split())
        print(f"    notes : {note_str}")


def _download_model(
    model: dict,
    volume_root: Path,
    *,
    force: bool,
    dry_run: bool,
    hf_token: Optional[str],
    civitai_token: Optional[str],
) -> tuple[str, str]:
    """Return (status, message) where status in ('ok', 'skip', 'error', 'dry')."""
    mid = model.get("id", "?")
    url = (model.get("url") or "").strip()
    filename = (model.get("filename") or "").strip()
    dest_rel = (model.get("dest") or "").strip()
    auth = model.get("auth") or ""

    if not url:
        return "skip", f"{mid}: url is not set in models.yaml"

    if not filename:
        return "error", f"{mid}: filename is not set in models.yaml"

    dest_abs = volume_root / dest_rel

    # Pick the right token based on 'auth' field and source.
    token_hf = hf_token if auth == "hf_token" or "huggingface.co" in url or "hf.co" in url else None
    token_civitai = civitai_token if "civitai.com" in url else None

    if dry_run:
        print(
            _colored("[DRY-RUN]", COLOR_YELLOW)
            + f" would download {_colored(filename, COLOR_BOLD)}"
            + f"\n  url  : {url}"
            + f"\n  dest : {dest_abs / filename}"
        )
        return "dry", mid

    print(_colored(f"\n[{mid}] {filename}", COLOR_BOLD))
    print(f"  dest: {dest_abs / filename}")

    try:
        download(
            url=url,
            output_dir=dest_abs,
            filename=filename,
            force=force,
            hf_token=token_hf,
            civitai_token=token_civitai,
        )
        return "ok", f"{mid}: {filename}"
    except FileExistsError as exc:
        return "skip", f"{mid}: {exc} (use --force to overwrite)"
    except Exception as exc:  # noqa: BLE001
        return "error", f"{mid}: {exc}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download all required models defined in scripts/models.yaml.",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST),
        help=f"Path to models.yaml (default: {MANIFEST})",
    )
    parser.add_argument(
        "--volume-root",
        default=None,
        help="Override volume root path (default: models.yaml → RUNPOD_VOLUME_DIR → /runpod-volume).",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        choices=["flux2_reactor", "wan22_i2v"],
        help="Download only models required by this workflow.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        metavar="ID",
        default=[],
        help="Skip model with this id (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making any network requests.",
    )
    parser.add_argument(
        "--init-dirs",
        action="store_true",
        help="Create the directory structure on the volume and exit (no downloads).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. Priority: this flag → $HF_TOKEN env → scripts/config.yaml.",
    )
    parser.add_argument(
        "--civitai-token",
        default=None,
        help="Civitai token. Priority: this flag → $CIVITAI_TOKEN env → scripts/config.yaml.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.stderr.write(f"ERROR: manifest not found: {manifest_path}\n")
        return 2

    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: failed to load manifest: {exc}\n")
        return 2

    volume_root = _resolve_volume_root(manifest, args.volume_root)
    dirs = manifest.get("directories") or []
    all_models: list[dict] = manifest.get("models") or []

    hf_token = resolve_hf_token(args.hf_token)
    civitai_token = resolve_civitai_token(args.civitai_token)

    print(_colored(f"Volume root: {volume_root}", COLOR_BOLD))
    print(f"Manifest   : {manifest_path}")

    # Always create directories (unless dry-run skips the fs ops).
    _create_dirs(volume_root, dirs, dry_run=args.dry_run)

    if args.init_dirs:
        print("\nDone (init-dirs).")
        return 0

    skip_ids = set(args.skip)
    models = _filter_models(all_models, workflow=args.workflow, skip_ids=skip_ids)

    if not models:
        print("\nNo models to download (after filtering).")
        return 0

    print(_colored(f"\nModels to download ({len(models)}):", COLOR_BOLD))
    for m in models:
        _print_summary_line(m, volume_root)

    if args.dry_run:
        print()
        for m in models:
            _download_model(
                m, volume_root,
                force=args.force,
                dry_run=True,
                hf_token=hf_token,
                civitai_token=civitai_token,
            )
        print("\nDone (dry-run).")
        return 0

    results: dict[str, list[str]] = {"ok": [], "skip": [], "error": []}

    for m in models:
        status, msg = _download_model(
            m, volume_root,
            force=args.force,
            dry_run=False,
            hf_token=hf_token,
            civitai_token=civitai_token,
        )
        results.setdefault(status, []).append(msg)

    # Summary
    print(_colored("\n" + "=" * 60, COLOR_BOLD))
    print(_colored("Summary", COLOR_BOLD))
    print(_colored("=" * 60, COLOR_BOLD))

    for msg in results.get("ok", []):
        print(f"  {_colored('✓ OK   ', COLOR_GREEN)} {msg}")
    for msg in results.get("skip", []):
        print(f"  {_colored('- SKIP ', COLOR_YELLOW)} {msg}")
    for msg in results.get("error", []):
        print(f"  {_colored('✗ ERROR', COLOR_RED)} {msg}")

    errors = results.get("error", [])
    if errors:
        print(_colored(f"\n{len(errors)} error(s). Check the messages above.", COLOR_RED))
        return 1

    print(_colored("\nAll done.", COLOR_GREEN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
