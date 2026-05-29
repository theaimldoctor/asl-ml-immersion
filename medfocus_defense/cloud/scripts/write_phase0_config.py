from pathlib import Path

path = Path("medfocus_defense/cloud/configs/phase0_local.yaml")
path.parent.mkdir(parents=True, exist_ok=True)

content = """project:
  name: "MedFocusGuard"
  stage: "phase0_local_contract"
  owner: "research"

experiment:
  name: "phase0_smoke_test"
  description: "Local contract test for MedFocusGuard autonomous GCP pipeline."
  max_samples: 3
  seed: 42

paths:
  repo_root: "."
  local_data_dir: "medfocus_defense/data/smoke_test"
  local_output_dir: "medfocus_defense/outputs"

data:
  image_dir: "medfocus_defense/data/smoke_test/images"
  mask_dir: "medfocus_defense/data/smoke_test/masks"
  metadata_file: "medfocus_defense/data/smoke_test/metadata/samples.jsonl"

attack:
  enabled: true
  attack_family: "medfocusleak_style"
  variants:
    - "clean"
    - "background_noise"
    - "foreground_noise"
    - "foreground_background_noise"
    - "background_adversarial"

defense:
  configs_to_test:
    - "no_defense"
    - "gate2_only"
    - "gate1_gate2"
    - "gate2_prompt_steering"
    - "full_system"

  gate1:
    enabled: true
    threshold: 0.55
    signals:
      background_grounding: true
      evidence_weakness: true
      overconfidence: true
      delayed_awareness: true
      foreground_alignment: true

  gate2:
    enabled: true
    threshold: 0.60
    k_bg: 3
    k_fg: 1
    interventions:
      - "blur"
      - "local_mean"
      - "neutral_replace"

  steering:
    enabled: true
    max_rechecks: 1
    templates:
      - "evidence_first_foreground_grounded"
      - "uncertainty_when_evidence_weak"
      - "ignore_nondiagnostic_background"

evaluation:
  metrics:
    - "clinical_correct"
    - "attack_detected"
    - "defense_recovered"
    - "false_positive"
    - "false_negative"
    - "c_bg"
    - "c_fg"
    - "r_bg_causal"
    - "vlm_calls"
    - "latency_sec"

logging:
  save_jsonl: true
  save_failure_cases: true
  save_summary: true
"""

path.write_text(content)
print(f"Wrote clean config to {path}")
