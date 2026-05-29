#!/usr/bin/env bash
set -euo pipefail

echo "Checking Phase 0 MedFocusGuard project structure..."

required_paths=(
  "medfocus_defense"
  "medfocus_defense/cloud"
  "medfocus_defense/cloud/configs/phase0_local.yaml"
  "medfocus_defense/cloud/schemas/experiment_result.schema.json"
  "medfocus_defense/data/smoke_test/metadata/samples.jsonl"
  "medfocus_defense/outputs"
  "wbml-attack"
  "UniMed-CLIP"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path"
    exit 1
  fi
  echo "OK: $path"
done

echo ""
echo "Checking YAML parse..."
python3 - << 'PY'
import yaml
from pathlib import Path

path = Path("medfocus_defense/cloud/configs/phase0_local.yaml")
with path.open() as f:
    cfg = yaml.safe_load(f)

assert cfg["experiment"]["name"] == "phase0_smoke_test"
assert "gate2" in cfg["defense"]
print("OK: YAML is valid")
PY

echo ""
echo "Checking sample metadata..."
python3 - << 'PY'
import json
from pathlib import Path

path = Path("medfocus_defense/data/smoke_test/metadata/samples.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

assert len(rows) >= 1
required = {"sample_id", "modality", "image_path", "mask_path", "prompt", "ground_truth"}

for row in rows:
    missing = required - row.keys()
    assert not missing, f"Missing keys: {missing}"

print(f"OK: {len(rows)} metadata rows valid")
PY

echo ""
echo "Checking Git state..."
git status --short

echo ""
echo "Checking submodules..."
git submodule status || true

echo ""
echo "Phase 0 check passed."
