"""Load the published BAT evaluation checkpoint after verifying its contents."""

import hashlib
import tempfile

import torch


# Hugging Face LFS SHA-256 for zhisheng01/SpatialAudio/SpatialAST/finetuned.pth.
BAT_CHECKPOINT_SHA256 = "b19bd06bb8961e2e6b9a716f198a0ec13d24ef23d8350ebb148772f2ba59645b"


def load_bat_checkpoint(path):
    digest = hashlib.sha256()
    # Deserialize the private verified copy, not a path that can change after hashing.
    with tempfile.TemporaryFile() as verified, open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            verified.write(chunk)
        if digest.hexdigest() != BAT_CHECKPOINT_SHA256:
            raise ValueError("Checkpoint does not match the published BAT evaluation checkpoint; download finetuned.pth from the README link.")
        verified.seek(0)
        return torch.load(verified, map_location="cpu")
