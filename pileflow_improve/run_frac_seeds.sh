#!/usr/bin/env bash
# Train (and optionally evaluate) the improved PileFlow for several seeds.
#
#   bash run_frac_seeds.sh /path/to/ppjj_scalar500_npu140_56k_v1.npz            # train 5 seeds
#   PUMML=/path/to/pumml_56k_seed1.pt bash run_frac_seeds.sh /path/to/data.npz  # + K=32 sampling + comparison
#   PUMML=runs/pumml_56k_seedSEED.pt bash run_frac_seeds.sh /path/to/data.npz   # PileFlow seed s vs PUMML seed s
#
# Run it in the background on the server so it survives logout:
#   nohup bash run_frac_seeds.sh /path/to/data.npz > run_frac_seeds.log 2>&1 &
#
# Optional environment variables:
#   SEEDS="1 2 3 4 5"   seeds to run
#   DEVICE=cuda         device (default: cuda if available)
#   K=32                posterior samples per test jet
#   EPOCHS=80           max training epochs (e.g. EPOCHS=2 for a quick test)
#   PUMML=...           PUMML checkpoint; if set, compare_methods.py is run for every seed.
#                       The word SEED in it is replaced by the seed, so seed s is paired with PUMML seed s.
set -euo pipefail

DATA=${1:?usage: bash run_frac_seeds.sh /path/to/ppjj_scalar500_npu140_56k_v1.npz}
SEEDS=${SEEDS:-"1 2 3 4 5"}
K=${K:-32}
EPOCHS=${EPOCHS:-80}
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
[ -f "$DATA" ] || { echo "Data file not found: $DATA"; exit 1; }
DATA=$(realpath "$DATA")                                  # absolute paths, since we cd below
if [ -n "${PUMML:-}" ]; then                             # absolute path(s), check they exist
  for s in $SEEDS; do f=${PUMML//SEED/$s}; [ -f "$f" ] || { echo "PUMML model not found: $f"; exit 1; }; done
  PUMML=$(realpath "$(dirname "$PUMML")")/$(basename "$PUMML")
fi
cd "$(dirname "$0")"
mkdir -p runs results
echo "Data: $DATA | seeds: $SEEDS | device: $DEVICE | K: $K"
[ "$DEVICE" = cuda ] || echo "WARNING: no GPU found, training will be slow (~50 min per seed on CPU)"

for s in $SEEDS; do
  out=runs/frac_seed$s
  echo "=== seed $s: training -> $out ($(date))"
  python -m pileflow.run --data "$DATA" --device "$DEVICE" --arch spatial_v2 --base-channels 48 --n-blocks 4 \
      --target-mode fraction --epochs "$EPOCHS" --patience 20 --seed "$s" \
      --eval-splits --outdir "$out" > "$out.train.log" 2>&1
  tail -n 2 "$out.train.log"

  if [ -n "${PUMML:-}" ]; then
    echo "=== seed $s: sampling test set with K=$K ($(date))"
    python evaluate_checkpoint.py --data "$DATA" --device "$DEVICE" --checkpoint "$out/checkpoints/pileflow_best.pt" \
        --split eval --eval-samples "$K" --ode-solver midpoint --ode-steps 8 --outdir "$out" --no-plots \
        > "$out.sample.log" 2>&1
    pumml_s=${PUMML//SEED/$s}
    echo "=== seed $s: comparison with PUMML $(basename "$pumml_s") ($(date))"
    python compare_methods.py --data "$DATA" --device "$DEVICE" --pumml-model "$pumml_s" \
        --pileflow "$out/data/generated_eval.npz" --add-posterior-obs \
        --outdir "results/test_frac_seed$s" > "$out.compare.log" 2>&1
    echo "    results in results/test_frac_seed$s"
  fi
done
if [ -n "${PUMML:-}" ]; then
  python aggregate_seeds.py $(for s in $SEEDS; do echo results/test_frac_seed$s; done) --outdir results/seeds_summary
fi
echo "All done ($(date))"
