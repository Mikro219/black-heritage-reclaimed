"""
setup_voice.py — Download and verify the Vosk small English model.

Run once before the first session:
    python scripts/setup_voice.py

The model is saved to models/vosk-model-small-en-us-0.15/ at the repo root.
That directory is gitignored — this script is the canonical way to get it.
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_DIR = MODELS_DIR / MODEL_NAME
MODEL_ZIP = MODELS_DIR / f"{MODEL_NAME}.zip"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
# SHA-256 of the upstream zip — recheck if alphacephei re-releases the file.
EXPECTED_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def main() -> None:
    print(f"Vosk model target: {MODEL_DIR}")

    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print("Model directory already exists and is non-empty — nothing to do.")
        print("If you want to re-download, delete the directory and run this script again.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    if MODEL_ZIP.exists():
        print(f"Zip already downloaded: {MODEL_ZIP}")
    else:
        print(f"Downloading {MODEL_URL} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_ZIP, _progress_hook)
            print()  # newline after progress
        except Exception as exc:
            print(f"\nDownload failed: {exc}", file=sys.stderr)
            MODEL_ZIP.unlink(missing_ok=True)
            sys.exit(1)

    # Checksum
    print("Verifying checksum ...", end=" ", flush=True)
    actual = _sha256(MODEL_ZIP)
    if actual != EXPECTED_SHA256:
        print(f"FAIL\n  expected: {EXPECTED_SHA256}\n  got:      {actual}", file=sys.stderr)
        print("The zip may be corrupted or the upstream file changed. Delete it and retry.")
        MODEL_ZIP.unlink(missing_ok=True)
        sys.exit(1)
    print("OK")

    # Extract
    print(f"Extracting to {MODELS_DIR} ...")
    with zipfile.ZipFile(MODEL_ZIP) as zf:
        zf.extractall(MODELS_DIR)

    # Clean up zip
    MODEL_ZIP.unlink()

    if not MODEL_DIR.exists():
        print(
            f"Extraction succeeded but expected directory not found: {MODEL_DIR}\n"
            "Check that the zip's top-level folder is named exactly '{MODEL_NAME}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Done. Model ready at: {MODEL_DIR}")
    print("Run 'python -m engines.voice_engine --test-scene 1' to verify keyword detection.")


if __name__ == "__main__":
    main()
