#!/usr/bin/env python3
"""
Convert the Vietnamese metadata CSV (audio_path|text|speaker_id) into the JSONL
manifest format expected by the preprocessing pipeline.

Usage:
    python tools/prepare_vi_metadata.py \
        --csv metadata.csv \
        --audio-root /mnt/datasets/vivoice \
        --output manifests/vi_raw.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert metadata CSV to JSONL manifest.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to metadata CSV (audio_path|text|speaker_id).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Directory containing the referenced audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL manifest path.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="|",
        help="Column delimiter used in the CSV file.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code to tag samples with.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_root = args.audio_root.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    if not audio_root.is_dir():
        raise NotADirectoryError(f"Audio root is not a directory: {audio_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8") as csv_file, output_path.open("w", encoding="utf-8") as jsonl_file:
        reader = csv.DictReader(csv_file, delimiter=args.delimiter)
        required_fields = {"audio_path", "text", "speaker_id"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing_fields))}")

        processed = 0
        skipped = 0

        for row in reader:
            if row is None or (len(row) == 1 and None in row):
                skipped += 1
                continue

            audio_value = (row.get("audio_path") or "").strip()
            text = (row.get("text") or "").strip()
            speaker = (row.get("speaker_id") or "").strip() or "spk_default"

            if not audio_value:
                skipped += 1
                continue

            relative_audio = Path(audio_value)

            if not text:
                skipped += 1
                continue

            audio_path = (audio_root / relative_audio).resolve()
            if not audio_path.is_file():
                skipped += 1
                continue

            record = {
                "id": relative_audio.stem,
                "audio": str(audio_path),
                "text": text,
                "speaker": speaker,
                "language": args.language,
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    print(f"[Done] Wrote {processed} samples to {output_path} (skipped {skipped}).")


if __name__ == "__main__":
    main()
