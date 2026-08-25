#!/bin/bash
# Restart run.sh when it dies, because the one identified risk on this box is a TRANSIENT
# failure: the DiT cells reserve ~2.7GB against ~4.2GB free, so a browser or video spike can
# CUDA-OOM a single run. run.sh exits 1 on any phase failure (correct -- never sweep on
# incomplete checkpoints), which would otherwise leave the GPU idle until a human noticed.
#
# Safe to loop only because every phase is resumable and existence of a result file means
# done: a retry re-runs exactly the run that died and nothing else.
#
# The progress guard is what stops this being a spin-loop. A transient OOM leaves artifacts
# from the runs that DID finish, so the count rises between attempts; a deterministic bug
# produces the same count twice and we stop rather than burn the night on it.
#
# It takes TWO consecutive no-progress failures to give up, not one. A transient OOM can land
# before any artifact does -- e.g. during the first training run of a phase -- and a one-strike
# guard would write that off as deterministic and refuse to retry something that would have
# worked. Caught by testing the supervisor rather than by reading it.
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
MAX=${MAX:-6}
BACKOFF=${BACKOFF:-60}
RUN=${RUN:-./run.sh}

count () { find out/ckpt out/res out/probe -type f 2>/dev/null | wc -l; }

prev=$(count)
stuck=0
echo "supervisor: starting, $prev artifacts on disk, max $MAX attempts"
for i in $(seq 1 "$MAX"); do
  if $RUN; then echo "supervisor: $RUN completed rc=0 after $i attempt(s)"; exit 0; fi
  now=$(count)
  if [ "$now" -le "$prev" ]; then stuck=$((stuck+1)); else stuck=0; fi
  echo "supervisor: attempt $i failed, artifacts $prev -> $now, no-progress streak $stuck"
  if [ "$stuck" -ge 2 ]; then
    echo "supervisor: two consecutive attempts made no progress -- deterministic, giving up"
    exit 1
  fi
  prev=$now
  echo "supervisor: retrying in ${BACKOFF}s"
  sleep "$BACKOFF"
done
echo "supervisor: exhausted $MAX attempts"; exit 1
