#!/bin/bash
# Phase driver for the napkin-dit sweep. Resumable at every level: existence of a result
# file means done, so re-running this after any interruption picks up the remainder.
# Each phase announces its own completion into $D/.done.<phase> with its exit status --
# watchers poll for that FILE and never for a process (Research Metastrategy #24-#33).
set -u
D=$(cd "$(dirname "$0")" && pwd)          # absolute: never rely on inherited cwd here
cd "$D"
PY=${PY:-python3}
STEPS=${STEPS:-14000}
# Concurrency is a per-MACHINE tuning, and the right value inverted between the two boxes
# this ran on. Measured, unet, 250 steps after warmup, GPU-resident data:
#   laptop 4050 (12 threads, 6GB):  NPAR=1 -> 14.7 st/s aggregate
#                                   NPAR=2 ->  6.4 st/s each = 12.9 aggregate  (WORSE)
#   Colab T4   (2 vCPU):            4 workers each pinned at ~43% of a core, i.e. the two
#                                   cores saturated and the GPU not -- concurrency helped.
# One process already saturates the 4050, so the default is 1. Override for a wide machine:
# NPAR=4 ./run.sh. Do not raise it blind -- measure, because the bottleneck moves.
NPAR=${NPAR:-1}
LOG=$D/logs; mkdir -p "$LOG" out

phase () {                                 # phase <name> <cmd...>
  local n=$1; shift
  if [ -f "$D/.done.$n" ] && [ "$(cat "$D/.done.$n")" = 0 ]; then
    echo "== $n already done"; return 0; fi
  echo "== $n starting $(date -u +%FT%TZ)"
  { "$@"; echo "$?" > "$D/.done.$n"; } >> "$LOG/$n.log" 2>&1
  local rc; read -r rc < "$D/.done.$n"
  echo "== $n rc=$rc $(date -u +%FT%TZ)"
  [ "$rc" = 0 ] || return 1
}

train_all () {                             # 20 runs, NPAR at a time
  for b in unet dit; do for o in eps flow; do for s in 0 1 2 3 4; do
    echo "$b $o $s"
  done; done; done | xargs -P "$NPAR" -L1 bash -c \
    '$0 napkin_dit.py train --backbone $1 --objective $2 --seed $3 --steps '"$STEPS"' \
       > logs/train-$1-$2-s$3.log 2>&1 || echo "FAILED $1 $2 $3"' "$PY"
}

sweep_sharded () {                         # one process per seed
  # NOTE the two kinds of $1 below. Outside the single quotes, "$1" is THIS function's
  # argument (the tier) and expands now; inside them, $1 is the inner shell's first
  # positional (the seed) and expands per xargs line. They look identical and are not, so
  # the tier is bound to a named local first -- a second reader read this as a bug.
  local tier=$1
  $PY napkin_dit.py sweep --tier "$tier" --seed 0 --nfe 2 || return 1   # builds out/clf.pt
  # -P "$NPAR", not a hardcoded 5: the sweep contends for the same CPU and the same GPU as
  # training does, so there is no reason for its concurrency to be a different number.
  for s in 0 1 2 3 4; do echo "$s"; done | xargs -P "$NPAR" -L1 bash -c \
    '$0 napkin_dit.py sweep --tier "$1" --seed "$2" > "logs/sweep-$1-s$2.log" 2>&1 \
       || echo "FAILED sweep $1 seed $2"' "$PY" "$tier"
}

phase selfcheck  $PY napkin_dit.py selfcheck            || exit 1
probe_all () {                             # shard by (backbone, lr), then reduce
  # Grid must BRACKET the optimum, not end at it. The first pass used {1e-4,2e-4,5e-4} and
  # both backbones picked 5e-4, the maximum -- with the DiT still improving 34% per grid step
  # there. cmd_probe now refuses to publish a boundary pick, so this grid extends upward.
  for b in unet dit; do for lr in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2; do echo "$b $lr"; done; done \
    | xargs -P "$NPAR" -L1 bash -c \
      '$0 napkin_dit.py probe --backbone $1 --lrs $2 --steps '"$STEPS"' \
         > logs/probe-$1-$2.log 2>&1' "$PY" || { echo "probe shards failed"; return 1; }
  # Delete FIRST, so the existence test below can only be satisfied by THIS run. Previously
  # it was `[ -f out/lr.json ]` against a file a PREVIOUS probe had written, which reported
  # success while every shard and the reduce had crashed (Metastrategy #28).
  rm -f out/lr.json
  $PY napkin_dit.py probe --steps "$STEPS" || { echo "probe reduce failed"; return 1; }
  [ -f out/lr.json ] || { echo "reduce wrote no lr.json (boundary refusal?)"; return 1; }
}

phase probe      probe_all                               || exit 1
phase train      train_all                               || exit 1
phase sweep_headline   sweep_sharded headline            || exit 1
phase agg_headline     $PY napkin_dit.py agg --tier headline
phase sweep_secondary  sweep_sharded secondary
phase sweep_secondary2 sweep_sharded secondary2
phase agg_secondary    $PY napkin_dit.py agg --tier secondary
phase gif        $PY napkin_dit.py gif
echo ALL-PHASES-COMPLETE
