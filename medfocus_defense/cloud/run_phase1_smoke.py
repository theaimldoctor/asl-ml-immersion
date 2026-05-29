import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from google.cloud import bigquery
from google.cloud import storage


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


def simulate_gate_scores(sample_idx: int, attack_variant: str, defense_config: str) -> dict[str, float]:
    """
    Phase 1 simulator.
    Later we replace this with real Gate 1/Gate 2 outputs.
    """
    base_bg = {
        "clean": 0.20,
        "background_noise": 0.45,
        "foreground_noise": 0.25,
        "foreground_background_noise": 0.55,
        "background_adversarial": 0.75,
    }[attack_variant]

    base_fg = {
        "clean": 0.20,
        "background_noise": 0.25,
        "foreground_noise": 0.60,
        "foreground_background_noise": 0.55,
        "background_adversarial": 0.25,
    }[attack_variant]

    jitter = 0.02 * sample_idx
    c_bg = min(max(base_bg + jitter, 0.0), 1.0)
    c_fg = min(max(base_fg - jitter, 0.0), 1.0)
    r_bg_causal = c_bg / (c_bg + c_fg + 1e-8)

    gate1_score = {
        "clean": 0.15,
        "background_noise": 0.42,
        "foreground_noise": 0.35,
        "foreground_background_noise": 0.58,
        "background_adversarial": 0.78,
    }[attack_variant]

    if defense_config == "no_defense":
        gate1_score = 0.0
        r_bg_causal = 0.0

    return {
        "gate1_score": gate1_score,
        "c_bg": c_bg,
        "c_fg": c_fg,
        "r_bg_causal": r_bg_causal,
        "gate2_score": r_bg_causal,
    }


def decide_outcome(
    attack_variant: str,
    defense_config: str,
    gate1_score: float,
    r_bg_causal: float,
    gate1_threshold: float,
    gate2_threshold: float,
) -> dict[str, Any]:
    attacked = attack_variant != "clean"

    if defense_config == "no_defense":
        attack_detected = False
        recovered = False
        final_decision = "accept_unchecked"
    elif defense_config == "gate1_gate2":
        gate1_triggered = gate1_score > gate1_threshold
        attack_detected = gate1_triggered and r_bg_causal > gate2_threshold
        recovered = attack_detected
        final_decision = "escalate" if attack_detected else "accept"
    else:
        attack_detected = r_bg_causal > gate2_threshold
        recovered = attack_detected and defense_config in {
            "gate2_only",
            "gate2_prompt_steering",
            "full_system",
        }
        final_decision = "escalate" if attack_detected else "accept"

    false_positive = (not attacked) and attack_detected
    false_negative = attacked and (not attack_detected)

    # For Phase 1: clinical_correct means the final system did not silently accept a harmful attacked case.
    clinical_correct = (not attacked) or recovered or final_decision == "escalate"

    return {
        "attack_detected": attack_detected,
        "defense_recovered": recovered,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "clinical_correct": clinical_correct,
        "final_decision": final_decision,
    }


def build_result_row(
    run_id: str,
    sample: dict[str, Any],
    sample_idx: int,
    attack_variant: str,
    defense_config: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    gate1_threshold = float(cfg["defense"]["gate1"]["threshold"])
    gate2_threshold = float(cfg["defense"]["gate2"]["threshold"])

    scores = simulate_gate_scores(sample_idx, attack_variant, defense_config)
    outcome = decide_outcome(
        attack_variant=attack_variant,
        defense_config=defense_config,
        gate1_score=scores["gate1_score"],
        r_bg_causal=scores["r_bg_causal"],
        gate1_threshold=gate1_threshold,
        gate2_threshold=gate2_threshold,
    )

    vlm_calls = {
        "no_defense": 1,
        "gate2_only": 1 + int(cfg["defense"]["gate2"]["k_bg"]) + int(cfg["defense"]["gate2"]["k_fg"]),
        "gate1_gate2": 1 + int(scores["gate1_score"] > gate1_threshold) * (
            int(cfg["defense"]["gate2"]["k_bg"]) + int(cfg["defense"]["gate2"]["k_fg"])
        ),
        "gate2_prompt_steering": 1 + int(cfg["defense"]["gate2"]["k_bg"]) + int(cfg["defense"]["gate2"]["k_fg"]) + 1,
        "full_system": 1 + int(cfg["defense"]["gate2"]["k_bg"]) + int(cfg["defense"]["gate2"]["k_fg"]) + 1,
    }[defense_config]

    latency_sec = round(0.4 * vlm_calls + random.uniform(0.02, 0.10), 3)

    return {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "sample_id": sample["sample_id"],
        "modality": sample.get("modality", "unknown"),
        "attack_type": "phase1_simulated_medfocusleak",
        "noise_location": attack_variant,
        "defense_config": defense_config,
        "gate1_enabled": defense_config == "gate1_gate2",
        "gate1_score": float(scores["gate1_score"]),
        "gate2_score": float(scores["gate2_score"]),
        "c_bg": float(scores["c_bg"]),
        "c_fg": float(scores["c_fg"]),
        "r_bg_causal": float(scores["r_bg_causal"]),
        "final_decision": outcome["final_decision"],
        "clinical_correct": bool(outcome["clinical_correct"]),
        "attack_detected": bool(outcome["attack_detected"]),
        "defense_recovered": bool(outcome["defense_recovered"]),
        "false_positive": bool(outcome["false_positive"]),
        "false_negative": bool(outcome["false_negative"]),
        "vlm_calls": int(vlm_calls),
        "latency_sec": float(latency_sec),
        "notes": "phase1 smoke simulation; replace with real VLM/Gate modules in Phase 2",
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def upload_to_gcs(project_id: str, bucket_name: str, local_path: Path, gcs_path: str) -> None:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def insert_bigquery_rows(project_id: str, dataset: str, table: str, rows: list[dict[str, Any]]) -> None:
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.{table}"
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


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

    samples = load_jsonl(
        cfg["paths"]["metadata_file"],
        int(cfg["experiment"]["max_samples"]),
    )

    output_dir = Path(cfg["paths"]["local_output_dir"]) / run_id
    ensure_dir(output_dir)

    rows = []
    start_time = time.time()

    for sample_idx, sample in enumerate(samples):
        for attack_variant in cfg["attack"]["variants"]:
            for defense_config in cfg["defense"]["configs_to_test"]:
                rows.append(
                    build_result_row(
                        run_id=run_id,
                        sample=sample,
                        sample_idx=sample_idx,
                        attack_variant=attack_variant,
                        defense_config=defense_config,
                        cfg=cfg,
                    )
                )

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    write_jsonl(rows, results_path)

    summary = {
        "run_id": run_id,
        "num_samples": len(samples),
        "num_rows": len(rows),
        "attack_variants": cfg["attack"]["variants"],
        "defense_configs": cfg["defense"]["configs_to_test"],
        "elapsed_sec": round(time.time() - start_time, 3),
        "local_results_path": str(results_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if cfg["logging"].get("upload_to_gcs", True):
        gcs_prefix = f"runs/{run_id}"
        upload_to_gcs(project_id, bucket, results_path, f"{gcs_prefix}/results.jsonl")
        upload_to_gcs(project_id, bucket, summary_path, f"{gcs_prefix}/summary.json")

    if cfg["logging"].get("insert_to_bigquery", True):
        insert_bigquery_rows(project_id, bq_dataset, bq_table, rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
