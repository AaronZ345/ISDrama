"""Extract and evaluate paired spatial audio features from the repository root."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=root / "data/gt")
    parser.add_argument("--pred-dir", type=Path, default=root / "data/infer")
    parser.add_argument("--checkpoint", type=Path, default=root / "ckpt/finetuned.pth")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-embeddings", action="store_true", help="Extract and evaluate IPD/ILD only.")
    parser.add_argument("--mae-normalization", choices=("mean", "legacy"), default="mean")
    args = parser.parse_args()
    gt_dir, pred_dir = args.gt_dir.resolve(), args.pred_dir.resolve()
    gt_names = {path.name for path in gt_dir.glob("*.wav") if path.is_file()}
    pred_names = {path.name for path in pred_dir.glob("*.wav") if path.is_file()}
    if not gt_names or gt_names != pred_names:
        parser.error("Ground-truth and prediction directories must contain the same nonempty set of WAV filenames.")
    if not args.skip_embeddings and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    output_root = args.output_dir.resolve()
    outputs = {kind: output_root / kind / "npy" for kind in ("gt", "infer")}
    for output_dir in outputs.values():
        if any(output_dir.glob("*.npy")):
            parser.error(f"Feature directory is not empty: {output_dir}. Choose a fresh --output-dir.")

    for kind, audio_dir in (("gt", gt_dir), ("infer", pred_dir)):
        output_dir = outputs[kind]
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_embeddings:
            subprocess.run([
                sys.executable, str(root / "main.py"),
                "--model", "build_EVA", "--dataset", "mrsaudio",
                "--audio_path_root", str(audio_dir),
                "--output_dir", str(output_dir),
                "--finetune", str(args.checkpoint.resolve()),
                "--device", args.device, "--nb_classes", "355",
                "--batch_size", "32", "--num_workers", "4",
                "--audio_normalize", "--eval",
            ], cwd=root, check=True)
        subprocess.run([
            sys.executable, str(root / "evaluate/gen_ipd_ild.py"),
            "--audio-dir", str(audio_dir), "--output-dir", str(output_dir),
        ], cwd=root, check=True)

    command = [
        sys.executable, str(root / "evaluate/evaluate.py"),
        "--gt-dir", str(outputs["gt"]), "--pred-dir", str(outputs["infer"]),
        "--mae-normalization", args.mae_normalization,
    ]
    if args.skip_embeddings:
        command.append("--skip-embeddings")
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
