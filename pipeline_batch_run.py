"""Canonical batch runner for the keyframe-centric AIC pipeline.

This file intentionally delegates to the maintained entrypoints instead of
duplicating training, feature loading, indexing, and remote push logic.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _run(args: list[str]) -> None:
    logger.info("Running: %s", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical AIC 2026 keyframe pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Limit videos/keyframes where supported (0=all)")
    parser.add_argument("--force-import", action="store_true", help="Regenerate per-video metadata JSONL")
    parser.add_argument("--with-transcript", action="store_true", help="Run Whisper transcript assignment during import")
    parser.add_argument("--translate-titles", action="store_true", help="Translate video titles during import")
    parser.add_argument("--captions", action="store_true", help="Generate BLIP-2 captions before training/indexing")
    parser.add_argument("--train-lora", action="store_true", help="Train LoRA CLIP before indexing")
    parser.add_argument("--epochs", type=int, default=3, help="LoRA epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training/caption batch size where supported")
    parser.add_argument("--num-workers", type=int, default=4, help="Training DataLoader workers")
    parser.add_argument("--push-remote", action="store_true", help="Push keyframe vectors to Qdrant after local index")
    parser.add_argument("--recreate-remote", action="store_true", help="Recreate Qdrant collection before push")
    args = parser.parse_args()

    import_cmd = ["scripts/import_btc_data.py", "--limit", str(args.limit)]
    if args.force_import:
        import_cmd.append("--force")
    if args.with_transcript:
        import_cmd.append("--with-transcript")
    if args.translate_titles:
        import_cmd.append("--translate-titles")
    _run(import_cmd)

    if args.captions:
        _run([
            "-m",
            "backend.preprocessing.generate_captions",
            "--batch-size",
            str(args.batch_size),
            "--limit",
            str(args.limit),
        ])

    if args.train_lora:
        _run([
            "scripts/train_lora_clip.py",
            "--limit",
            str(args.limit),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ])

    _run(["-m", "backend.embedding.build_index", "--limit", str(args.limit)])

    if args.push_remote:
        remote_cmd = ["-m", "backend.embedding.push_to_remote"]
        if args.recreate_remote:
            remote_cmd.append("--recreate")
        _run(remote_cmd)

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
