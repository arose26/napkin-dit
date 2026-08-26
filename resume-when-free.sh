#!/bin/bash
# Wait until the GPU has room, then resume the sweep.
#
# Why: napkin_skips.py started training on the same 6GB card mid-sweep. Combined VRAM hit
# 5641/6141 MiB and NFE-63 sweep points went from 384s to >5200s. WSL spills to shared host
# memory instead of OOMing, so the failure mode is a 13x slowdown rather than a crash -- which
# means supervise.sh, which only retries on FAILURE, would never have noticed.
#
# The condition is FREE VRAM, deliberately not "is napkin_skips still running". Keying on a
# process pattern was tried and rejected: `pgrep -f` matches any shell whose argv contains the
# pattern -- a wrapper script, or one of my own diagnostic commands -- so the watcher can see a
# job that already exited and wait forever. Testing it caught exactly that (Metastrategy #25,
# #33). Free VRAM measures the resource we actually need and cannot be fooled by argv.
#
#   ./resume-when-free.sh
#   NEED_MIB=2600 POLL=120 ./resume-when-free.sh
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
NEED_MIB=${NEED_MIB:-2600}      # DiT training peak ~2.2GB reserved + ~0.5GB CUDA context
POLL=${POLL:-120}
STABLE=${STABLE:-3}             # consecutive clear checks, so a momentary dip cannot trigger
RESUME=${RESUME:-./supervise.sh}
NVSMI=${NVSMI:-nvidia-smi}

free_mib () {
  local used total
  read -r used total < <($NVSMI --query-gpu=memory.used,memory.total \
                          --format=csv,noheader,nounits | tr -d ',' | head -1)
  echo $(( total - used ))
}

echo "resume-watcher: need ${NEED_MIB}MiB free for ${STABLE} consecutive checks, polling ${POLL}s"
ok=0; n=0
while [ "$ok" -lt "$STABLE" ]; do
  f=$(free_mib)
  if [ "$f" -ge "$NEED_MIB" ]; then
    ok=$((ok+1)); echo "resume-watcher: ${f}MiB free ($ok/$STABLE) $(date -u +%FT%TZ)"
  else
    [ "$ok" -gt 0 ] && echo "resume-watcher: dipped to ${f}MiB, resetting"
    ok=0; n=$((n+1))
    [ $((n % 15)) = 1 ] && echo "resume-watcher: ${f}MiB free, need ${NEED_MIB} $(date -u +%FT%TZ)"
  fi
  [ "$ok" -lt "$STABLE" ] && sleep "$POLL"
done
echo "resume-watcher: resuming $RESUME at $(date -u +%FT%TZ)"
exec $RESUME
