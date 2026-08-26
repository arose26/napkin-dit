#!/bin/bash
# The resume watcher's whole job is to NOT fire at the wrong moment, and its first version got
# the threshold wrong in exactly that direction. Stubbed nvidia-smi, no GPU needed.
#   ./test_resume.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); fails=0
t=$(mktemp -d); cp "$HERE/resume-when-free.sh" "$t/"
printf '#!/bin/bash\necho RESUMED\n' > "$t/fake.sh"; chmod +x "$t/fake.sh"
mk () { printf '#!/bin/bash\necho "%s"\n' "$1" > "$t/nv"; chmod +x "$t/nv"; }
check () {                                  # check <name> <want_fire yes|no>
  local name=$1 want=$2 out rc
  out=$(cd "$t" && POLL=1 STABLE=2 RESUME=./fake.sh NVSMI=./nv timeout 6 ./resume-when-free.sh 2>&1); rc=$?
  local got=no; echo "$out" | grep -q RESUMED && got=yes
  if [ "$got" = "$want" ]; then echo "  ok   $name (fired=$got)"
  else echo "  FAIL $name: fired=$got want=$want"; fails=$((fails+1)); fi
}
mk "3180, 6141"; check "other job running (2961 free) must NOT fire" no
mk "1968, 6141"; check "idle (4173 free) must fire"                  yes
mk "5641, 6141"; check "both jobs (500 free) must NOT fire"          no
mk "";           check "nvidia-smi returns nothing must NOT fire"    no
mk "garbage";    check "nvidia-smi returns junk must NOT fire"       no
rm -rf "$t"
[ "$fails" = 0 ] && echo "test_resume OK" || { echo "test_resume: $fails failed"; exit 1; }
