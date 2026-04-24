#!/usr/bin/env python3
"""Download a model file from HuggingFace, Civitai, or a direct URL.

Usage
-----

    python scripts/download_model.py <URL> <OUTPUT_DIR> [--filename NAME] [--force]
    python scripts/download_model.py --init-dirs [--volume-root /runpod-volume]

The destination path is ``<OUTPUT_DIR>/<filename>`` where ``<filename>`` is
either ``--filename`` or auto-detected from the URL / Content-Disposition
header. The script prints a progress bar during the download and performs
an atomic rename at the end so partial files are never left behind.

Supported sources
-----------------

* **Direct URL** — any URL that returns the file bytes.
* **HuggingFace** — standard ``https://huggingface.co/<repo>/resolve/<rev>/<path>``
  URLs work out of the box. A token can be supplied via ``--hf-token`` or the
  ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env variable for gated / private
  repositories.
* **Civitai** — ``https://civitai.com/api/download/models/<id>`` (or any
  ``civitai.com`` URL). A token can be supplied via ``--civitai-token`` or
  the ``CIVITAI_TOKEN`` env variable. The redirect target is followed
  automatically and the filename is taken from the ``Content-Disposition``
  header when available.

Dependencies
------------

* ``requests``
* ``tqdm``

Install with::

    pip install requests tqdm

Examples
--------

::

    # HuggingFace
    python scripts/download_model.py \
        https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/flux2_dev_fp8mixed.safetensors \
        /runpod-volume/models/diffusion_models

    # Civitai (requires CIVITAI_TOKEN for most models)
    CIVITAI_TOKEN=xxx python scripts/download_model.py \
        https://civitai.com/api/download/models/123456 \
        /runpod-volume/models/loras \
        --filename portrait_style_lora.safetensors

    # Direct URL
    python scripts/download_model.py \
        https://example.com/path/to/model.safetensors \
        /runpod-volume/models/custom
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

try:
    import requests
except ImportError as exc:  # pragma: no cover
    sys.stderr.write("ERROR: 'requests' is required. Install with: pip install requests tqdm\n")
    raise SystemExit(1) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    sys.stderr.write("ERROR: 'tqdm' is required. Install with: pip install requests tqdm\n")
    raise SystemExit(1) from exc

from config_loader import resolve_civitai_token, resolve_hf_token


CHUNK_SIZE = 1024 * 1024  # 1 MiB
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300


def _is_huggingface(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("huggingface.co") or host.endswith("hf.co")


def _is_civitai(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("civitai.com")


def _build_headers(url: str, hf_token: Optional[str], civitai_token: Optional[str]) -> dict:
    headers = {"User-Agent": "runpod-serverless-generation-downloader/1.0"}
    if _is_huggingface(url) and hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    elif _is_civitai(url) and civitai_token:
        # Civitai also accepts ?token=... query param, but Authorization works too.
        headers["Authorization"] = f"Bearer {civitai_token}"
    return headers


_FILENAME_STAR_RE = re.compile(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;]+)", re.IGNORECASE)
_FILENAME_RE = re.compile(r'filename\s*=\s*"?([^";]+)"?', re.IGNORECASE)


def _filename_from_content_disposition(value: str) -> Optional[str]:
    # Prefer RFC 5987 filename* (may be percent-encoded).
    m = _FILENAME_STAR_RE.search(value)
    if m:
        return unquote(m.group(1).strip().strip('"'))
    m = _FILENAME_RE.search(value)
    if m:
        return m.group(1).strip()
    return None


def _filename_from_url(url: str) -> Optional[str]:
    path = urlparse(url).path
    if not path:
        return None
    name = os.path.basename(path)
    if not name:
        return None
    name = unquote(name)
    # Avoid picking something meaningless like "download" or a bare number.
    if "." not in name:
        return None
    return name


def _resolve_filename(
    response: requests.Response,
    url: str,
    explicit: Optional[str],
) -> str:
    if explicit:
        return explicit
    cd = response.headers.get("Content-Disposition")
    if cd:
        name = _filename_from_content_disposition(cd)
        if name:
            return name
    # Use the URL of the final response (after redirects), then the original URL.
    for candidate_url in (response.url, url):
        name = _filename_from_url(candidate_url)
        if name:
            return name
    raise RuntimeError(
        "Could not determine output filename. Pass --filename explicitly."
    )


def _human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def download(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    *,
    force: bool = False,
    hf_token: Optional[str] = None,
    civitai_token: Optional[str] = None,
) -> Path:
    """Download ``url`` into ``output_dir``. Returns the final file path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = _build_headers(url, hf_token=hf_token, civitai_token=civitai_token)

    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    ) as response:
        if response.status_code == 401:
            raise RuntimeError(
                f"HTTP 401 Unauthorized for {url}. "
                "Provide --hf-token / HF_TOKEN or --civitai-token / CIVITAI_TOKEN."
            )
        if response.status_code == 403:
            raise RuntimeError(
                f"HTTP 403 Forbidden for {url}. The token may lack access to this resource."
            )
        response.raise_for_status()

        final_name = _resolve_filename(response, url, filename)
        # Safety: never allow path traversal in the auto-detected filename.
        if "/" in final_name or "\\" in final_name or final_name in ("", ".", ".."):
            raise RuntimeError(f"Unsafe filename detected: {final_name!r}")

        final_path = output_dir / final_name
        if final_path.exists() and not force:
            raise FileExistsError(
                f"{final_path} already exists. Pass --force to overwrite."
            )

        total_str = response.headers.get("Content-Length")
        total = int(total_str) if total_str and total_str.isdigit() else None

        sys.stderr.write(f"Downloading: {url}\n")
        sys.stderr.write(f"  -> {final_path}\n")
        if total is not None:
            sys.stderr.write(f"  size: {_human_size(total)} ({total} bytes)\n")
        else:
            sys.stderr.write("  size: unknown\n")
        sys.stderr.flush()

        # Write to a temp file in the same directory, then atomically rename.
        fd, tmp_name = tempfile.mkstemp(
            prefix=final_name + ".",
            suffix=".part",
            dir=str(output_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            with open(tmp_path, "wb") as out, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                miniters=1,
                desc=final_name,
            ) as bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    out.write(chunk)
                    bar.update(len(chunk))

            if total is not None:
                actual = tmp_path.stat().st_size
                if actual != total:
                    raise RuntimeError(
                        f"Downloaded size {actual} does not match Content-Length {total}."
                    )

            if final_path.exists() and force:
                final_path.unlink()
            os.replace(tmp_path, final_path)
        except BaseException:
            # Clean up partial file on any failure (including KeyboardInterrupt).
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    sys.stderr.write(f"Done: {final_path}\n")
    return final_path


# Standard volume subdirectories created by --init-dirs.
_VOLUME_DIRS = [
    "models/diffusion_models",
    "models/text_encoders",
    "models/vae",
    "models/insightface",
    "models/loras",
    "models/checkpoints",
    "models/custom",
    "storage/inputs",
    "storage/images",
    "storage/videos",
    "storage/previews",
    "storage/metadata",
    "tmp/generation",
]


def init_volume_dirs(volume_root: Path) -> None:
    """Create the full directory structure required by the worker."""
    for rel in _VOLUME_DIRS:
        d = volume_root / rel
        d.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(f"  mkdir: {d}\n")
    sys.stderr.write("Done.\n")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a model file from HuggingFace, Civitai, or a direct URL.",
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Source URL (HuggingFace / Civitai / direct). Omit when using --init-dirs.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory to save the file into (will be created if it does not exist).",
    )
    parser.add_argument(
        "--init-dirs",
        action="store_true",
        help="Create the full volume directory structure and exit (no download).",
    )
    parser.add_argument(
        "--volume-root",
        default=os.environ.get("RUNPOD_VOLUME_DIR", "/runpod-volume"),
        help="Volume root used by --init-dirs (default: $RUNPOD_VOLUME_DIR or /runpod-volume).",
    )
    parser.add_argument(
        "--filename",
        help="Override the output filename. If omitted, it is derived from "
             "Content-Disposition or the URL path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
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
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.init_dirs:
        root = Path(args.volume_root)
        sys.stderr.write(f"Initialising volume directory structure under {root}\n")
        init_volume_dirs(root)
        return 0

    if not args.url or not args.output_dir:
        sys.stderr.write(
            "ERROR: url and output_dir are required unless --init-dirs is used.\n"
            "Run with -h for help.\n"
        )
        return 2

    hf_token = resolve_hf_token(args.hf_token)
    civitai_token = resolve_civitai_token(args.civitai_token)

    try:
        download(
            url=args.url,
            output_dir=Path(args.output_dir),
            filename=args.filename,
            force=args.force,
            hf_token=hf_token,
            civitai_token=civitai_token,
        )
    except FileExistsError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    except requests.HTTPError as exc:
        sys.stderr.write(f"HTTP error: {exc}\n")
        return 1
    except requests.RequestException as exc:
        sys.stderr.write(f"Network error: {exc}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
