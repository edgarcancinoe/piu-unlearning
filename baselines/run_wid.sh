#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash baselines/run_wid.sh IDENTITY_ID [IR_SE50_CHECKPOINT] [--data-dir DIR] [--output-dir DIR] [--use-anchor-overrides]" >&2
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if (($# < 1)); then usage; exit 2; fi
identity_id=$1
shift
checkpoint=""
if (($#)) && [[ $1 != --* ]]; then checkpoint=$1; shift; fi
data_dir=data/celebahq_512
output_dir="outputs/wid_$identity_id"
use_anchor_overrides=false

while (($#)); do
  case $1 in
    --data-dir) data_dir=${2:?Missing directory after --data-dir}; shift 2 ;;
    --output-dir) output_dir=${2:?Missing directory after --output-dir}; shift 2 ;;
    --use-anchor-overrides) use_anchor_overrides=true; shift ;;
    *) usage; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
if [[ -n "$checkpoint" && ! -f "$checkpoint" ]]; then echo "IR-SE50 checkpoint not found: $checkpoint" >&2; exit 1; fi
command -v piu-prepare-data >/dev/null || { echo "Install the package first; piu-prepare-data is not on PATH." >&2; exit 1; }

required=(embeddings.npy centroids.npy centroid_labels.npy)
found=0
missing=0
for file in "${required[@]}"; do
  if [[ -f "$data_dir/$file" ]]; then ((found += 1)); else ((missing += 1)); fi
done
if [[ -f "$data_dir/srk_labels_eps0.35.npy" || -f "$data_dir/labels.npy" ]]; then ((found += 1)); else ((missing += 1)); fi
if ((found == 0)); then
  piu-prepare-data --output-dir "$data_dir"
elif ((missing > 0)); then
  echo "Prepared data is incomplete in $data_dir; repair it before running WID." >&2
  exit 1
fi

wid_args=""
if [[ -n "$checkpoint" ]]; then wid_args=$(python -c 'import shlex,sys; print(shlex.join(["--identity-checkpoint", sys.argv[1]]))' "$checkpoint"); fi
launcher_args=(--identity-id "$identity_id" --methods wid --data-dir "$data_dir" --output-dir "$output_dir")
if [[ $use_anchor_overrides == true ]]; then launcher_args+=(--use-anchor-overrides); fi
python baselines/run_all.py "${launcher_args[@]}" "--wid-args=$wid_args"
