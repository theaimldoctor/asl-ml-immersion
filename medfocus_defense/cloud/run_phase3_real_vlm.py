import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import yaml
from PIL import Image, ImageFilter
from google.cloud import bigquery, storage

from medfocus_defense.cloud.vlm_backends import get_vlm_backend


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str, max_samples: int) -> list[dict]:
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


def invert_mask(mask: Image.Image) -> Image.Image:
    arr = np.array(mask)
    return Image.fromarray(255 - arr, mode="L")


def apply_region(original: Image.Image, modified: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(modified, original, mask)


def blur_region(img: Image.Image, mask: Image.Image, radius: int = 7) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return apply_region(img, blurred, mask)


def local_mean_region(img: Image.Image, mask: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    mask_arr = np.array(mask) > 127

    out = arr.copy()
    if mask_arr.any():
        mean_val = arr[mask_arr].mean(axis=0)
        out[mask_arr] = mean_val

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def neutral_replace_region(img: Image.Image, mask: Image.Image, value: int = 128) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    mask_arr = np.array(mask) > 127

    out = arr.copy()
    out[mask_arr] = np.array([value, value, value], dtype=np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def make_intervention(img: Image.Image, fg_mask: Image.Image, region: str, intervention: str) -> Image.Image:
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


def text_change_delta(a: str, b: str) -> float:
    sim = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 1.0 - sim


def save_jsonl(rows: list[dict], path: Path) -> None:
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


def insert_bigquery_rows(project_id: str, dataset: str, table: str, rows: list[dict]) -> None:
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.{table}"
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


def run_gate2_for_sample(run_id: str, sample: dict, cfg: dict, output_dir: Path, backend) -> dict:
    sample_id = sample["sample_id"]
    sample_dir = output_dir / sample_id
    ensure_dir(sample_dir)

    image = load_image(sample["image_path"])

    if not sample.get("mask_path"):
        raise ValueError(f"Sample {sample_id} has no mask_path. Generate masks first.")

    fg_mask = load_mask(sample["mask_path"], invert=bool(cfg["mask"]["invert_mask"]))

    original_path = sample_dir / "original.png"
    mask_path = sample_dir / "foreground_mask.png"

    image.save(original_path)
    fg_mask.save(mask_path)

    prompt = cfg["victim"]["prompt"]
    y0 = backend.generate(str(original_path), prompt)

    bg_deltas = []
    fg_deltas = []
    bg_reports = {}
    fg_reports = {}

    for intervention in cfg["gate2"]["interventions"]["background"]:
        edited = make_intervention(image, fg_mask, "background", intervention)
        edited_path = sample_dir / f"bg_{intervention}.png"
        edited.save(edited_path)

        yj = backend.generate(str(edited_path), prompt)
        delta = text_change_delta(y0, yj)
        bg_deltas.append(delta)

        bg_reports[intervention] = {
            "image": str(edited_path),
            "report": yj,
            "delta_from_original": delta,
        }

    for intervention in cfg["gate2"]["interventions"]["foreground"]:
        edited = make_intervention(image, fg_mask, "foreground", intervention)
        edited_path = sample_dir / f"fg_{intervention}.png"
        edited.save(edited_path)

        yj = backend.generate(str(edited_path), prompt)
        delta = text_change_delta(y0, yj)
        fg_deltas.append(delta)

        fg_reports[intervention] = {
            "image": str(edited_path),
            "report": yj,
            "delta_from_original": delta,
        }

    c_bg = float(np.mean(bg_deltas)) if bg_deltas else 0.0
    c_fg = float(np.mean(fg_deltas)) if fg_deltas else 0.0
    r_bg_causal = c_bg / (c_bg + c_fg + 1e-8)

    threshold = float(cfg["gate2"]["threshold"])
    attack_detected = r_bg_causal > threshold
    final_decision = "escalate" if attack_detected else "accept"

    report_bundle = {
        "sample_id": sample_id,
        "prompt": prompt,
        victim_cfg = cfg.get("victim", {})
        backend_name = victim_cfg.get("backend", "simple_image")

        backend_kwargs = {
            k: v for k, v in victim_cfg.items()
            if k not in {"backend", "prompt"}
       }

        backend = get_vlm_backend(backend_name, **backend_kwargs),
        "original_report": y0,
        "background_reports": bg_reports,
        "foreground_reports": fg_reports,
        "c_bg": c_bg,
        "c_fg": c_fg,
        "r_bg_causal": r_bg_causal,
        "threshold": threshold,
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
        "attack_type": "phase3_real_vlm_backend_test",
        "noise_location": "background_and_foreground_interventions",
        "defense_config": f"gate2_{cfg['victim']['backend']}",
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
            "phase": "phase3",
            "backend": cfg["victim"]["backend"],
            "sample_output_dir": str(sample_dir),
        }),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    project_id = cfg["project"]["project_id"]
    bucket = cfg["project"]["bucket"]
    bq_dataset = cfg["project"]["bq_dataset"]
    bq_table = cfg["project"]["bq_table"]

    backend = get_vlm_backend(cfg["victim"]["backend"])

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

    for sample in samples:
        row_start = time.time()
        row = run_gate2_for_sample(run_id, sample, cfg, output_dir, backend)
        row["latency_sec"] = round(time.time() - row_start, 3)
        rows.append(row)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    save_jsonl(rows, results_path)

    summary = {
        "run_id": run_id,
        "phase": "phase3",
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
