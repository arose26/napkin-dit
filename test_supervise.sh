#!/bin/bash
# The supervisor decides whether a failure is worth retrying, so its loop is exactly the kind
# of code that fails silently in the direction of "did nothing all night". Four cases, each
# run against a stub instead of the real pipeline.
#
# Case 4 is the one that matters and the one interactive testing surfaced: a transient failure
# that lands BEFORE any artifact does. A one-strike progress guard calls that deterministic
# and refuses to retry a run that would have worked.
#
#   ./test_supervise.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
fails=0

run_case () {                              # run_case <name> <want_rc> <want_attempts> <stub>
  local name=$1 want_rc=$2 want_att=$3 stub=$4
  local t; t=$(mktemp -d)
  cp "$HERE/supervise.sh" "$t/"; mkdir -p "$t/out/ckpt" "$t/out/res" "$t/out/probe"
  printf '%s\n' "$stub" > "$t/stub.sh"; chmod +x "$t/stub.sh"
  ( cd "$t" && MAX=6 BACKOFF=1 RUN=./stub.sh ./supervise.sh >/dev/null 2>&1 )
  local rc=$? att; att=$(cat "$t/.attempt" 2>/dev/null || echo 0)
  if [ "$rc" = "$want_rc" ] && [ "$att" = "$want_att" ]; then
    echo "  ok   $name (rc=$rc attempts=$att)"
  else
    echo "  FAIL $name: rc=$rc want $want_rc, attempts=$att want $want_att"; fails=$((fails+1))
  fi
  rm -rf "$t"
}

N='N=$(cat .attempt 2>/dev/null || echo 0); N=$((N+1)); echo $N > .attempt'

# transient, each attempt lands an artifact -> retries until it passes
run_case "transient with progress" 0 3 "#!/bin/bash
$N
touch out/ckpt/a-\$N.pt; [ \$N -ge 3 ] && exit 0; exit 1"

# transient, first attempt produces NOTHING -> must still retry (the case that fixed the guard)
run_case "transient, no artifact yet" 0 2 "#!/bin/bash
$N
[ \$N -ge 2 ] && { touch out/ckpt/a.pt; exit 0; }; exit 1"

# deterministic -> must stop at two strikes, not burn all MAX attempts
run_case "deterministic failure" 1 2 "#!/bin/bash
$N
exit 1"

# clean pass -> must not retry at all
run_case "clean pass" 0 1 "#!/bin/bash
$N
exit 0"

[ "$fails" = 0 ] && echo "test_supervise OK" || { echo "test_supervise: $fails failed"; exit 1; }
