#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST_ONLY=0

if [[ "${1:-}" == "--list-only" ]]; then
  LIST_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--list-only]"
  exit 1
fi

frameworks=(keras pytorch)
datasets=(cifar10 cifar100)
models=(resnet50 mobilenetv2)
precisions=(fp32)
seeds=(42 123 456)

commands=()

for framework in "${frameworks[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      for precision in "${precisions[@]}"; do
        for seed in "${seeds[@]}"; do
          if [[ "$framework" == "keras" ]]; then
            script_path="$ROOT_DIR/src/keras/train.py"
            cmd=(python "$script_path" --dataset "$dataset" --model "$model" --precision "$precision" --seed "$seed")
          else
            script_path="$ROOT_DIR/src/pytorch/train.py"
            cmd=(python "$script_path" --dataset "$dataset" --model "$model" --precision "$precision" --seed "$seed")
            if [[ -n "${NUM_WORKERS:-}" ]]; then
              cmd+=(--num_workers "$NUM_WORKERS")
            fi
          fi

          printf -v cmd_str "%q " "${cmd[@]}"
          commands+=("${cmd_str% }")
        done
      done
    done
  done
done

echo "stage1_total_commands=${#commands[@]}"
for i in "${!commands[@]}"; do
  printf "[%02d/%02d] %s\n" "$((i + 1))" "${#commands[@]}" "${commands[$i]}"
done

if [[ "${#commands[@]}" -ne 24 ]]; then
  echo "ERROR: expected 24 Stage 1 commands, got ${#commands[@]}"
  exit 1
fi

if [[ "$LIST_ONLY" -eq 1 ]]; then
  exit 0
fi

for i in "${!commands[@]}"; do
  printf "running [%02d/%02d]\n" "$((i + 1))" "${#commands[@]}"
  eval "${commands[$i]}"
done
