#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_run_stats.sh RUN_DIR [OUTPUT_ZIP]

Create a zip archive containing run statistics/artifacts only. Model weights and
checkpoints are excluded.

Included by extension:
  .json .csv .tsv .png .jpg .jpeg .svg .pdf .txt .log .md .yaml .yml

Always excluded by extension:
  .eqx .pt .pth .ckpt .safetensors .onnx .pkl .pickle .joblib .npy .npz .zip

Examples:
  scripts/export_run_stats.sh runs/suite_spectral_gating_jax_clean_all_datasets-20260519-044126
  scripts/export_run_stats.sh runs/my_run /tmp/my_run_stats.zip
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

run_dir="${1%/}"
if [[ ! -d "$run_dir" ]]; then
  echo "error: RUN_DIR does not exist or is not a directory: $run_dir" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  output_zip="$2"
else
  output_zip="${run_dir}-stats.zip"
fi

python3 - "$run_dir" "$output_zip" <<'PY'
from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
output_zip = Path(sys.argv[2]).resolve()

include_exts = {
    ".json",
    ".csv",
    ".tsv",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".txt",
    ".log",
    ".md",
    ".yaml",
    ".yml",
}
exclude_exts = {
    ".eqx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
    ".zip",
}
exclude_dirs = {
    "__pycache__",
    ".ipynb_checkpoints",
    "checkpoints",
    "checkpoint",
    "weights",
    "models",
}

if output_zip.exists() and output_zip.is_dir():
    raise SystemExit(f"error: OUTPUT_ZIP is a directory: {output_zip}")
output_zip.parent.mkdir(parents=True, exist_ok=True)

files: list[Path] = []
skipped_weight_files = 0
skipped_other_files = 0

for root, dirs, names in os.walk(run_dir):
    dirs[:] = [name for name in dirs if name not in exclude_dirs]
    for name in names:
        path = Path(root) / name
        suffix = path.suffix.lower()
        if suffix in exclude_exts:
            skipped_weight_files += 1
            continue
        if suffix in include_exts:
            files.append(path)
        else:
            skipped_other_files += 1

files.sort()
if not files:
    raise SystemExit(f"error: no statistics files found under {run_dir}")

base = run_dir.name
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_run_dir": str(run_dir),
    "archive_root": base,
    "included_file_count": len(files),
    "skipped_excluded_extension_count": skipped_weight_files,
    "skipped_other_extension_count": skipped_other_files,
    "included_extensions": sorted(include_exts),
    "excluded_extensions": sorted(exclude_exts),
}

tmp_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
if tmp_zip.exists():
    tmp_zip.unlink()

with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.writestr(f"{base}/stats_export_manifest.json", json.dumps(manifest, indent=2) + "\n")
    for path in files:
        arcname = Path(base) / path.relative_to(run_dir)
        zf.write(path, arcname.as_posix())

tmp_zip.replace(output_zip)
size_mb = output_zip.stat().st_size / (1024 * 1024)
print(f"wrote {output_zip}")
print(f"included {len(files)} files; skipped {skipped_weight_files} checkpoint/weight-like files")
print(f"archive size: {size_mb:.2f} MiB")
PY
