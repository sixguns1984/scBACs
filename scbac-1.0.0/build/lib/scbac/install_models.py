"""Download and manage the released scBAC pretrained model archive."""

from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

from .constants import (
    MODEL_ARCHIVE_NAME,
    MODEL_ROOT_NAME,
    PRETRAINED_STAGE_DIRS,
    ZENODO_MODEL_URL,
)
from .paths import get_model_root


REQUIRED_DIRS = sorted(set(PRETRAINED_STAGE_DIRS.values())) + ["benchmarking_model"]


def model_installation_complete(model_root):
    root = Path(model_root)
    return root.is_dir() and all((root / name).is_dir() for name in REQUIRED_DIRS)


def model_status(model_dir=None):
    root = get_model_root(model_dir)
    return {
        "model_root": str(root),
        "installed": model_installation_complete(root),
        "required_directories": REQUIRED_DIRS,
        "present_directories": [name for name in REQUIRED_DIRS if (root / name).is_dir()],
        "missing_directories": [name for name in REQUIRED_DIRS if not (root / name).is_dir()],
    }


def _download(url, destination, quiet=False, retries=3):
    """Stream the Zenodo archive with small retry protection."""
    destination = Path(destination)
    last_error = None
    for attempt in range(1, int(retries) + 1):
        try:
            try:
                import requests
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0) or 0)
                    done = 0
                    last = -1
                    with destination.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            done += len(chunk)
                            if not quiet and total:
                                pct = int(done * 100 / total)
                                if pct != last and pct % 5 == 0:
                                    print("  download: {}%".format(pct), flush=True)
                                    last = pct
                return
            except ImportError:
                from urllib.request import urlopen
                with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
                    total = int(response.headers.get("content-length", 0) or 0)
                    done = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        if not quiet and total:
                            print("\r  download: {:3d}%".format(int(done * 100 / total)), end="", flush=True)
                    if not quiet and total:
                        print()
                return
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            if attempt < retries:
                if not quiet:
                    print("  download attempt {}/{} failed: {}. Retrying...".format(attempt, retries, exc))
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Failed to download scBAC model archive after {} attempts: {}".format(retries, last_error))


def _locate_extracted_model_root(extract_dir):
    extract_dir = Path(extract_dir)
    direct = extract_dir / MODEL_ROOT_NAME
    if direct.is_dir():
        return direct
    if all((extract_dir / name).is_dir() for name in REQUIRED_DIRS):
        return extract_dir
    candidates = [p for p in extract_dir.rglob(MODEL_ROOT_NAME) if p.is_dir()]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in REQUIRED_DIRS):
            return candidate
    raise RuntimeError(
        "The downloaded archive was extracted, but its scBAC model root could not be identified."
    )


def install_pretrained_models(model_dir=None, force=False, quiet=False, url=None):
    """Download, extract and validate the released model archive."""
    target = get_model_root(model_dir)
    if model_installation_complete(target) and not force:
        if not quiet:
            print("scBAC pretrained models already installed at {}".format(target))
        return target

    url = url or os.environ.get("SCBAC_MODEL_URL", ZENODO_MODEL_URL)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("Downloading scBAC pretrained models from Zenodo")
        print("  URL:", url)
        print("  Destination:", target)

    with tempfile.TemporaryDirectory(prefix="scbac_models_") as tmp:
        tmp = Path(tmp)
        archive = tmp / MODEL_ARCHIVE_NAME
        _download(url, archive, quiet=quiet)
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        if not quiet:
            print("Extracting model archive...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)
        source_root = _locate_extracted_model_root(extract_dir)

        if target.exists():
            if force:
                shutil.rmtree(target)
            else:
                # A partial previous install is replaced atomically below.
                shutil.rmtree(target)
        shutil.copytree(source_root, target)

    if not model_installation_complete(target):
        missing = model_status(target)["missing_directories"]
        raise RuntimeError("Model installation incomplete; missing: {}".format(missing))

    marker = {
        "source_url": url,
        "installed_at_unix": time.time(),
        "model_root": str(target),
    }
    with (target / ".scbac_model_installation.json").open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2)

    if not quiet:
        print("Model installation complete:", target)
    return target


def ensure_pretrained_models(model_dir=None, auto_download=True, quiet=False):
    root = get_model_root(model_dir)
    if model_installation_complete(root):
        return root
    if not auto_download:
        raise FileNotFoundError(
            "scBAC pretrained models are not installed at {}. Run `scbac models install`.".format(root)
        )
    return install_pretrained_models(model_dir=root, force=False, quiet=quiet)


def build_parser():
    parser = argparse.ArgumentParser(description="Install scBAC pretrained model files.")
    parser.add_argument("--model-dir", default=None, help="Custom model root directory.")
    parser.add_argument("--force-download", "--force", action="store_true", dest="force")
    parser.add_argument("--list-models", action="store_true", help="Show installation status only.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(json.dumps(model_status(args.model_dir), indent=2))
        return 0
    install_pretrained_models(args.model_dir, force=args.force, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
