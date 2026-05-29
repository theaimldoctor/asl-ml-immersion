#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-avid-glazing-495107-i4}"
BQ_DATASET="${BQ_DATASET:-medfocusguard_research}"
BQ_TABLE="${BQ_TABLE:-experiment_metrics}"

bq query --use_legacy_sql=false "
SELECT
  run_id,
  defense_config,
  noise_location,
  COUNT(*) AS n,
  AVG(r_bg_causal) AS avg_r_bg_causal,
  AVG(CAST(attack_detected AS INT64)) AS detection_rate,
  AVG(CAST(false_negative AS INT64)) AS false_negative_rate,
  AVG(vlm_calls) AS avg_vlm_calls,
  AVG(latency_sec) AS avg_latency_sec
FROM \`${PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}\`
GROUP BY run_id, defense_config, noise_location
ORDER BY run_id DESC, defense_config, noise_location
LIMIT 50
"
