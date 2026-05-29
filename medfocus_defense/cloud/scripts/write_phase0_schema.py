from pathlib import Path

path = Path("medfocus_defense/cloud/schemas/experiment_result.schema.json")
path.parent.mkdir(parents=True, exist_ok=True)

content = """{
  "type": "object",
  "required": [
    "run_id",
    "timestamp_utc",
    "sample_id",
    "defense_config",
    "attack_type",
    "final_decision"
  ],
  "properties": {
    "run_id": {"type": "string"},
    "timestamp_utc": {"type": "string"},
    "sample_id": {"type": "string"},
    "modality": {"type": "string"},
    "attack_type": {"type": "string"},
    "noise_location": {"type": "string"},
    "defense_config": {"type": "string"},
    "gate1_enabled": {"type": "boolean"},
    "gate1_score": {"type": "number"},
    "gate2_score": {"type": "number"},
    "c_bg": {"type": "number"},
    "c_fg": {"type": "number"},
    "r_bg_causal": {"type": "number"},
    "final_decision": {"type": "string"},
    "clinical_correct": {"type": "boolean"},
    "attack_detected": {"type": "boolean"},
    "defense_recovered": {"type": "boolean"},
    "false_positive": {"type": "boolean"},
    "false_negative": {"type": "boolean"},
    "vlm_calls": {"type": "integer"},
    "latency_sec": {"type": "number"},
    "notes": {"type": "string"}
  }
}
"""

path.write_text(content)
print(f"Wrote schema to {path}")
