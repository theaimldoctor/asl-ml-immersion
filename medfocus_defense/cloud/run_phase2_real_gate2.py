import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageFilter
from google.cloud import bigquery, storage


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str, invert: bool = False) -> Image.Image:
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    binary = (arr > 127).astype(np.uint8) * 255
    if invert:
        binary = 255 - binary
    return Image.fromarray(binary, mode="L")


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_region(original: Image.Image, modified: Image.Image, mask: Image.Image) -> Image.Image:
    """
    mask white means apply modified region.
    """
    return Image.composite(modified, original, mask)


def blur_region(img: Image.Image, mask: Image.Image, radius: int = 7) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return apply_region(img, blurred, mask)


def local_mean_region(img: Image.Image, mask: Image.Image) -> Image.Image:
    arr = pil_to_np(img)
    mask_arr = np.array(mask) > 127

    out = arr.copy()
    if mask_arr.any():
        mean_val = arr[mask_arr].mean(axis=0)
        out[mask_arr] = mean_val

    return np_to_pil(out)


def neutral_replace_region(img: Image.Image, mask: Image.Image, value: int = 128) -> Image.Image:
    arr = pil_to_np(img)
    mask_arr = np.array(mask) > 127

    out = arr.copy()
    out[mask_arr] = np.array([value, value, value], dtype=np.float32)

    return np_to_pil(out)


def invert_mask(mask: Image.Image) -> Image.Image:
    arr = np.array(mask)
    return Image.fromarray(255 - arr, mode="L")


def make_intervention(
    img: Image.Image,
    fg_mask: Image.Image,
    region: str,
    intervention: str,
) -> Image.Image:
    if region == "foreground":
        region_mask = fg_mask
    elif region == "background":
        region_mask = invert_mask(fg_mask)
    else:
        raise ValueError(f"Unknown region: {region}")

    if intervention == "blur":
        return blur_region(img, region_mask)
    if intervention == "local_mean":
        return local_mean_region(img, region_mask)
    if intervention == "neutral_replace":
        return neutral_replace_region(img, region_mask)

    raise ValueError(f"Unknown intervention: {intervention}")


def simple_text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def victim_stub(image_path: str, prompt: str) -> str:
    """
    Temporary victim model stub.
    It produces deterministic text based on image brightness.
    Phase 3 will replace this with real VLM inference.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img).astype(np.float32)
    mean_intensity = arr.mean()
    std_intensity = arr.std()

    if mean_intensity > 125 and std_intensity < 55:
        finding = "The image appears mostly stable with no obvious focal abnormality."
    elif mean_intensity > 125:
        finding = "The image shows mild suspicious intensity variation near the clinical region."
    else:
        finding = "The image shows darker abnormal-appearing regions requiring clinical review."

    return f"Evidence first: {finding} Impression: cautious assessment recommended."


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def upload_file(project_id: str, bucket_name: str, local_path: Path, gcs_path: str) -> None:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def upload_directory(project_id: str, bucket_name: str, local_dir: Path, gcs_prefix: str) -> None:
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir)
            upload_file(project_id, bucket_name, path, f"{gcs_prefix}/{rel}")


def insert_bigquery_rows(project_id: str, dataset: str, table: str, rows: list[dict[str, Any]]) -> None:
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.{table}"
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


def run_gate2_for_sample(
    run_id: str,
    sample: dict[str, Any],
    sample_idx: int,
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    sample_dir = output_dir / sample_id
    ensure_dir(sample_dir)

    image = load_image(sample["image_path"])
    fg_mask = load_mask(sample["mask_path"], invert=bool(cfg["mask"]["invert_mask"]))

    original_path = sample_dir / "original.png"
    mask_path = sample_dir / "foreground_mask.png"

    image.save(original_path)
    fg_mask.save(mask_path)

    prompt = cfg["victim"]["prompt"]
    y0 = victim_stub(str(original_path), prompt)

    bg_deltas = []
    fg_deltas = []

    bg_reports = {}
    fg_reports = {}

    for intervention in cfg["gate2"]["interventions"]["background"]:
        edited = make_intervention(image, fg_mask, region="background", intervention=intervention)
        edited_path = sample_dir / f"bg_{intervention}.png"
        edited.save(edited_path)

        yj = victim_stub(str(edited_path), prompt)
        delta = 1.0 - simple_text_similarity(y0, yj)

        bg_deltas.append(delta)
        bg_reports[intervention] = yj

    for intervention in cfg["gate2"]["interventions"]["foreground"]:
        edited = make_intervention(image, fg_mask, region="foreground", intervention=intervention)
        edited_path = sample_dir / f"fg_{intervention}.png"
        edited.save(edited_path)

        yj = victim_stub(str(edited_path), prompt)
        delta = 1.0 - simple_text_similarity(y0, yj)

        fg_deltas.append(delta)
        fg_reports[intervention] = yj

    c_bg = float(np.mean(bg_deltas)) if bg_deltas else 0.0
    c_fg = float(np.mean(fg_deltas)) if fg_deltas else 0.0
    r_bg_causal = c_bg / (c_bg + c_fg + 1e-8)

    threshold = float(cfg["gate2"]["threshold"])
    attack_detected = r_bg_causal > threshold
    final_decision = "escalate" if attack_detected else "accept"

    report_bundle = {
        "sample_id": sample_id,
        "prompt": prompt,
        "original_report": y0,
        "background_reports": bg_reports,
        "foreground_reports": fg_reports,
        "c_bg": c_bg,
        "c_fg": c_fg,
        "r_bg_causal": r_bg_causal,
        "final_decision": final_decision,
    }

    (sample_dir / "gate2_report_bundle.json").write_text(
        json.dumps(report_bundle, indent=2),
        encoding="utf-8",
    )

    vlm_calls = 1 + len(bg_deltas) + len(fg_deltas)

    return {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "sample_id": sample_id,
        "modality": sample.get("modality", "unknown"),
        "attack_type": "phase2_real_intervention_test",
        "noise_location": "background_and_foreground_interventions",
        "defense_config": "gate2_real_interventions",
        "gate1_enabled": False,
        "gate1_score": 0.0,
        "gate2_score": r_bg_causal,
        "c_bg": c_bg,
        "c_fg": c_fg,
        "r_bg_causal": r_bg_causal,
        "final_decision": final_decision,
        "clinical_correct": not attack_detected,
        "attack_detected": attack_detected,
        "defense_recovered": attack_detected,
        "false_positive": False,
        "false_negative": False,
        "vlm_calls": vlm_calls,
        "latency_sec": 0.0,
        "notes": json.dumps({
            "phase": "phase2",
            "backend": cfg["victim"]["backend"],
            "sample_output_dir": str(sample_dir),
            "warning": "victim_stub used; replace with real VLM in Phase 3",
        }),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    random.seed(int(cfg["experiment"]["seed"]))

    project_id = cfg["project"]["project_id"]
    bucket = cfg["project"]["bucket"]
    bq_dataset = cfg["project"]["bq_dataset"]
    bq_table = cfg["project"]["bq_table"]

    run_id = (
        f"{cfg['experiment']['name']}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    output_dir = Path(cfg["paths"]["local_output_dir"]) / run_id
    ensure_dir(output_dir)

    samples = load_jsonl(
        cfg["paths"]["metadata_file"],
        int(cfg["experiment"]["max_samples"]),
    )

    start = time.time()
    rows = []

    for sample_idx, sample in enumerate(samples):
        row = run_gate2_for_sample(run_id, sample, sample_idx, cfg, output_dir)
        row["latency_sec"] = round(time.time() - start, 3)
        rows.append(row)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    save_jsonl(rows, results_path)

    summary = {
        "run_id": run_id,
        "phase": "phase2",
        "num_samples": len(samples),
        "num_rows": len(rows),
        "local_output_dir": str(output_dir),
        "elapsed_sec": round(time.time() - start, 3),
        "backend": cfg["victim"]["backend"],
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if cfg["logging"].get("upload_to_gcs", True):
        upload_directory(project_id, bucket, output_dir, f"runs/{run_id}")

    if cfg["logging"].get("insert_to_bigquery", True):
        insert_bigquery_rows(project_id, bq_dataset, bq_table, rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
