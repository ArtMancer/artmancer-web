#!/usr/bin/env python3
"""
Dataset preprocessing utilities for ArtMancer benchmark runner.

This script normalizes benchmark datasets that contain the following folders:
- input / with_object
- mae / mae_output (optional)
- mask
- target / groundtruth

It sorts files deterministically, re-aligns pairs by normalized filenames,
validates image dimensions, and copies them into a clean structure that the
benchmark CLI expects (input/, mask/, groundtruth/, mae/).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_preprocessor")

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

FOLDER_VARIANTS: Dict[str, List[str]] = {
    "input": ["input", "inputs", "with_object", "with-object", "withobject", "original", "source"],
    "mae": ["mae", "mae_output", "mae-images", "mae_output_images"],
    "mask": ["mask", "masks", "mask_images"],
    "target": ["target", "targets", "groundtruth", "ground_truth", "gt", "output"],
}


def natural_sort_key(path: Path) -> List[object]:
    """Return a key that sorts strings in human-friendly order."""
    pattern = re.compile(r"(\d+)")
    parts = pattern.split(path.stem.lower())
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    key.append(path.suffix.lower())
    return key


def normalize_stem(path: Path) -> str:
    """Normalize filename stem by removing common suffixes/prefixes."""
    stem = path.stem.lower()
    stem = re.sub(r"(?:_)?(input|mask|mae|target|gt|withobject|with_object|wo)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem or path.stem.lower()


def list_images(folder: Path) -> List[Path]:
    """List image files in a folder with deterministic ordering."""
    files: List[Path] = []
    for ext in ALLOWED_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    # Remove duplicates and sort
    unique_files = sorted({f.resolve() for f in files if f.is_file()}, key=natural_sort_key)
    return unique_files


def align_by_name(reference: Sequence[Path], candidates: Sequence[Path], label: str) -> List[Path]:
    """
    Try to align candidate files to reference order using normalized stems.
    Falls back to positional mapping if normalization fails.
    """
    mapping: Dict[str, List[Path]] = {}
    for candidate in candidates:
        mapping.setdefault(normalize_stem(candidate), []).append(candidate)

    ordered: List[Path] = []
    fallback_needed = False
    for ref in reference:
        key = normalize_stem(ref)
        bucket = mapping.get(key)
        if bucket:
            ordered.append(bucket.pop(0))
        else:
            fallback_needed = True
            break

    if fallback_needed or len(ordered) != len(reference):
        logger.warning(
            "⚠️  Unable to align %s files by normalized names. Falling back to positional order.", label
        )
        return list(candidates)

    return ordered


def ensure_same_length(data: Dict[str, List[Path]]) -> None:
    """Ensure all required folders contain the same number of images."""
    lengths = {key: len(files) for key, files in data.items() if key != "mae"}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Image count mismatch:\n" + "\n".join(f"  - {k}: {v}" for k, v in lengths.items())
        )


def validate_dimensions(
    input_files: Sequence[Path],
    mask_files: Sequence[Path],
    target_files: Sequence[Path],
    mae_files: Optional[Sequence[Path]] = None,
) -> None:
    """Validate that all paired images share the same spatial dimensions."""
    for idx, (inp, mask, tgt) in enumerate(zip(input_files, mask_files, target_files), start=1):
        with Image.open(inp) as img_in, Image.open(mask) as img_mask, Image.open(tgt) as img_target:
            in_size = img_in.size
            mask_size = img_mask.size
            tgt_size = img_target.size

        if in_size != tgt_size:
            raise ValueError(
                f"Input/target size mismatch at pair {idx}: input={in_size}, target={tgt_size}"
            )
        if mask_size != tgt_size:
            raise ValueError(
                f"Mask size mismatch at pair {idx}: mask={mask_size}, target={tgt_size}"
            )

    if mae_files:
        for idx, (inp, mae) in enumerate(zip(input_files, mae_files), start=1):
            with Image.open(inp) as img_in, Image.open(mae) as img_mae:
                if img_in.size != img_mae.size:
                    raise ValueError(
                        f"MAE size mismatch at pair {idx}: input={img_in.size}, mae={img_mae.size}"
                    )


def resolve_dataset_root(source: Path) -> Tuple[Path, Dict[str, Path]]:
    """
    Resolve dataset root and the actual folder paths for input/mae/mask/target.
    Searches the provided root and one level below.
    """
    candidates = [source]
    if source.is_dir():
        candidates.extend([p for p in source.iterdir() if p.is_dir()])

    for candidate in candidates:
        resolved: Dict[str, Path] = {}
        for key, variants in FOLDER_VARIANTS.items():
            for variant in variants:
                folder = candidate / variant
                if folder.exists() and folder.is_dir():
                    resolved[key] = folder
                    break
        # Require input/mask/target. mae is optional.
        if {"input", "mask", "target"}.issubset(resolved.keys()):
            return candidate, resolved

    raise FileNotFoundError(
        f"Could not locate dataset structure containing {list(FOLDER_VARIANTS.keys())} inside {source}"
    )


def copy_with_new_names(files: Sequence[Path], dest: Path) -> List[str]:
    """Copy files to destination directory with zero-padded sequential names."""
    dest.mkdir(parents=True, exist_ok=True)
    new_names: List[str] = []
    for idx, src in enumerate(files, start=1):
        ext = src.suffix.lower()
        new_name = f"{idx:05d}{ext}"
        shutil.copy2(src, dest / new_name)
        new_names.append(new_name)
    return new_names


@dataclass
class PreprocessResult:
    dataset_path: Path
    total_pairs: int
    manifest_path: Path
    mae_available: bool


def preprocess_dataset(source: Path, destination: Path, manifest_path: Path) -> PreprocessResult:
    """Main preprocessing entry point."""
    extraction_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        if source.is_file() and source.suffix.lower() == ".zip":
            extraction_dir = tempfile.TemporaryDirectory(prefix="benchmark_zip_")
            logger.info("📦 Extracting ZIP dataset to %s", extraction_dir.name)
            with zipfile.ZipFile(source, "r") as zip_ref:
                zip_ref.extractall(extraction_dir.name)
            working_root = Path(extraction_dir.name)
        else:
            working_root = source

        dataset_root, folders = resolve_dataset_root(working_root)
        logger.info("📁 Resolved dataset root: %s", dataset_root)
        logger.info(
            "🔍 Found folders - input: %s | mae: %s | mask: %s | target: %s",
            folders.get("input"),
            folders.get("mae", "N/A"),
            folders.get("mask"),
            folders.get("target"),
        )

        input_files = list_images(folders["input"])
        if not input_files:
            raise ValueError(f"No images found in input folder: {folders['input']}")
        mask_files = list_images(folders["mask"])
        target_files = list_images(folders["target"])
        mae_files = list_images(folders["mae"]) if "mae" in folders else []

        ensure_same_length({"input": input_files, "mask": mask_files, "target": target_files})

        # Align folders to input ordering
        mask_files = align_by_name(input_files, mask_files, "mask")
        target_files = align_by_name(input_files, target_files, "target")
        if mae_files:
            mae_files = align_by_name(input_files, mae_files, "mae")

        validate_dimensions(input_files, mask_files, target_files, mae_files or None)

        # Prepare destination structure
        if destination.exists():
            shutil.rmtree(destination)
        (destination / "input").mkdir(parents=True, exist_ok=True)
        (destination / "mask").mkdir(parents=True, exist_ok=True)
        (destination / "groundtruth").mkdir(parents=True, exist_ok=True)
        if mae_files:
            (destination / "mae").mkdir(parents=True, exist_ok=True)

        manifest_records: List[Dict[str, str]] = []

        new_input_names = copy_with_new_names(input_files, destination / "input")
        new_mask_names = copy_with_new_names(mask_files, destination / "mask")
        new_target_names = copy_with_new_names(target_files, destination / "groundtruth")
        new_mae_names: List[str] = []
        if mae_files:
            new_mae_names = copy_with_new_names(mae_files, destination / "mae")

        for idx in range(len(new_input_names)):
            record = {
                "index": idx + 1,
                "input_original": str(input_files[idx]),
                "input_formatted": new_input_names[idx],
                "mask_original": str(mask_files[idx]),
                "mask_formatted": new_mask_names[idx],
                "target_original": str(target_files[idx]),
                "target_formatted": new_target_names[idx],
            }
            if new_mae_names:
                record["mae_original"] = str(mae_files[idx])
                record["mae_formatted"] = new_mae_names[idx]
            manifest_records.append(record)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump({"pairs": manifest_records}, manifest_file, ensure_ascii=False, indent=2)

        logger.info("✅ Prepared %d paired samples", len(new_input_names))
        return PreprocessResult(
            dataset_path=destination,
            total_pairs=len(new_input_names),
            manifest_path=manifest_path,
            mae_available=bool(new_mae_names),
        )
    finally:
        if extraction_dir is not None:
            extraction_dir.cleanup()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize ArtMancer benchmark datasets.")
    parser.add_argument("--source", "-s", type=str, required=True, help="Path to dataset folder or ZIP file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Destination directory for normalized dataset (will be created).",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        type=str,
        default="manifest.json",
        help="Path to save manifest JSON (default: manifest.json inside output parent).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not source.exists():
        logger.error("❌ Source path does not exist: %s", source)
        sys.exit(1)

    try:
        result = preprocess_dataset(source, destination, manifest_path)
    except Exception as exc:
        logger.exception("❌ Failed to preprocess dataset: %s", exc)
        sys.exit(1)

    summary = {
        "dataset_path": str(result.dataset_path),
        "total_pairs": result.total_pairs,
        "manifest": str(result.manifest_path),
        "mae_available": result.mae_available,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


