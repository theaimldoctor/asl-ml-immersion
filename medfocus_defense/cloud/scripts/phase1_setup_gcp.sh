#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-avid-glazing-495107-i4}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT_ID}-medfocusguard}"
BQ_DATASET="${BQ_DATASET:-medfocusguard_research}"
BQ_TABLE="${BQ_TABLE:-experiment_metrics}"

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Bucket: gs://${BUCKET}"
echo "BigQuery: ${BQ_DATASET}.${BQ_TABLE}"

gcloud config set project "${PROJECT_ID}"

echo ""
echo "Enabling required APIs..."
gcloud services enable \
  storage.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com

echo ""
echo "Creating Cloud Storage bucket if needed..."
if gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "Bucket already exists: gs://${BUCKET}"
else
  gcloud storage buckets create "gs://${BUCKET}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
fi

echo ""
echo "Creating logical GCS prefixes..."
tmpfile="$(mktemp)"
echo "init" > "${tmpfile}"

gcloud storage cp "${tmpfile}" "gs://${BUCKET}/datasets/init.txt"
gcloud storage cp "${tmpfile}" "gs://${BUCKET}/configs/init.txt"
gcloud storage cp "${tmpfile}" "gs://${BUCKET}/runs/init.txt"
gcloud storage cp "${tmpfile}" "gs://${BUCKET}/reports/init.txt"

rm -f "${tmpfile}"

echo ""
echo "Creating BigQuery dataset if needed..."
if bq show --dataset "${PROJECT_ID}:${BQ_DATASET}" >/dev/null 2>&1; then
  echo "Dataset already exists: ${BQ_DATASET}"
else
  bq --location="${REGION}" mk --dataset "${PROJECT_ID}:${BQ_DATASET}"
fi

echo ""
echo "Phase 1 GCP setup completed."
