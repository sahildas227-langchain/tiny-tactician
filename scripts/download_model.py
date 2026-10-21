"""Download and verify the SmolLM2-360M-Instruct GGUF weights into ./models/."""

import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "QuantFactory/SmolLM2-360M-Instruct-GGUF"
FILENAME = "SmolLM2-360M-Instruct.Q4_K_M.gguf"
DST = Path(__file__).resolve().parents[1] / "models" / "smollm2-360m-instruct-q4_k_m.gguf"


def main() -> int:
    cached = Path(hf_hub_download(REPO, FILENAME))
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, DST)
    print("SRC", cached, cached.stat().st_size)
    print("DST", DST, DST.stat().st_size)
    print("SHA256", hashlib.sha256(DST.read_bytes()).hexdigest())
    if DST.read_bytes()[:4] != b"GGUF":
        print("ERROR: not a GGUF file")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
