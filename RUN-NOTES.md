# Run notes

Watchers do not survive a context refresh or a machine restart, so the arming commands live
here rather than being reconstructed (Research Metastrategy #32).

## Where it runs

Colab free tier, **Tesla T4 15360 MiB, 2 vCPU, 12GB RAM, torch 2.11.0+cu128**.
Notebook: `Untitled13.ipynb` (drive `1mTQINY81YilgklP1d_FxK0rTiaN4_pM_`), account K.
Working copy `/content/napkin-dit`, cloned from GitHub — code is never pasted through the
browser, the cell just pulls.

Not the laptop: the 6GB 4050 is held by series 3 (`napkin-nemesis`), and the local estimate
was ~15h of training while fighting a live experiment for SMs.

## Launch (idempotent — safe to run twice)

```bash
cd /content/napkin-dit
git pull -q
mkdir -p logs
NPAR=4 STEPS=14000 setsid nohup ./run.sh > driver.log 2>&1 &
```

**Never `git pull` while the driver is running.** bash reads a script incrementally, so
rewriting `run.sh` underneath a live driver can make it resume at a byte offset that is now
the middle of a different line. Pull only when no driver is running — i.e. before the launch,
or after a `.done.*` sentinel shows the last phase finished.

`setsid` matters: without it the driver dies with the cell. A `%%bash` cell **raises** on any
nonzero exit, so do not end such a cell with `ls .done.*` — with no sentinels yet that `ls`
exits 2 and the cell reports a failure while the detached driver is running perfectly well.

## Status poll (read by screenshot — Colab cell outputs are not scrapable over CDP)

```python
import subprocess
print(subprocess.run("cd /content/napkin-dit && tail -14 driver.log; "
  "for f in .done.*; do [ -f \"$f\" ] && echo \"  $f=$(cat $f)\"; done; "
  "echo \"probe=$(ls out/probe 2>/dev/null|wc -l)/6 ckpt=$(ls out/ckpt 2>/dev/null|wc -l)/20 "
  "res=$(ls out/res 2>/dev/null|wc -l)\"; "
  "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
  shell=True, capture_output=True, text=True).stdout)
```

Liveness is **file counts increasing between two ticks**, never "is a process alive"
(Metastrategy #27). `probe/`, `ckpt/` and `res/` counts are the three progress meters;
`.done.<phase>` files carry the phase's exit status, so `rc=7` is distinguishable from
`rc=0` and from "still going".

## Phases and their sentinels

| sentinel | what it covers | expected |
|---|---|---|
| `.done.selfcheck` | the whole assert suite on this device | ~6 min on T4 |
| `.done.probe` | 6 shards = 2 backbones x 3 LRs at 3500 steps, 4-wide | writes `out/lr.json` |
| `.done.train` | 20 runs (4 cells x 5 seeds) at 14000 steps, 4-wide | 20 files in `out/ckpt/` |
| `.done.sweep_headline` | heun x karras x 6 NFE x 5 seeds, sharded per seed | 120 files in `out/res/` |
| `.done.agg_headline` | IQM + bootstrap CI + rank stability | `out/agg-headline.json`, `out/ablation-headline.png` |
| `.done.sweep_secondary`, `.done.sweep_secondary2` | the spacing/solver rungs | more `out/res/` |
| `.done.gif` | 2x2 contact sheet | `out/denoise2x2.gif` |

## Retrieval — verify at the DESTINATION, never by the sending call's exit code

`files.download` has silently dropped artifacts before (a checkpoint and a gif, lost when a
tab closed and the runtime recycled). So: tar the small stuff, download **one file**, and
confirm it exists in `/mnt/c/Users/axelr/Downloads` from WSL before treating it as saved.
Treat the Colab VM as already dead the moment the run ends.

```python
!cd /content/napkin-dit && tar czf /content/napkin-dit-out.tgz out/res out/agg-*.json out/lr.json out/probe out/*.png out/*.gif 2>/dev/null; ls -la /content/napkin-dit-out.tgz
from google.colab import files; files.download('/content/napkin-dit-out.tgz')
```

The 20 checkpoints (~11MB each) stay on the VM unless needed — everything the writeup needs
is in `out/res/` and the two rendered artifacts.

## If the session dies

Re-run the launch block. Every phase and every result file is existence-checked, so a
restart costs at most the runs that were in flight. Colab free tier is one GPU per account
quota; the fallback account is `?authuser=1` as a **query** param (a `#...&authuser=1`
fragment is ignored).
