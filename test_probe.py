"""The one thing in the probe reducer that can silently corrupt a result: reading the wrong
shards. If it globbed the directory instead of the requested grid, a leftover shard from an
earlier --lrs set would win on stale numbers and get baked into out/lr.json for every run.

train_one is stubbed out, so this is pure bookkeeping -- no GPU, runs in a second.
    python3 test_probe.py
"""
import json, pathlib, shutil, tempfile, types
import napkin_dit as N


def shard(d, name, eps, flow):
    (d / "probe" / f"{name}.json").write_text(json.dumps({"eps": eps, "flow": flow}))


def main():
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "probe").mkdir()
    N.OUT = tmp
    N.train_one = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("trained a shard that already existed"))
    a = types.SimpleNamespace(backbone=["unet"], objective=["eps", "flow"],
                              lrs=[1e-4, 2e-4], bs=128, steps=8)

    # A leftover shard from a 4-LR grid, with the best losses on the board by a mile.
    shard(tmp, "unet-0.001", 0.01, 0.01)
    shard(tmp, "unet-0.0001", 0.5, 0.9)
    shard(tmp, "unet-0.0002", 0.2, 0.3)
    N.cmd_probe(a)
    best = json.loads((tmp / "lr.json").read_text())
    assert best == {"unet": 2e-4}, f"stale shard leaked into the ranking: picked {best}"

    # Mean rank across objectives, not a single objective's winner.
    shutil.rmtree(tmp / "probe"); (tmp / "probe").mkdir()
    shard(tmp, "unet-0.0001", 0.9, 0.9)      # loses both
    shard(tmp, "unet-0.0002", 0.2, 0.3)      # wins both
    N.cmd_probe(a)
    assert json.loads((tmp / "lr.json").read_text()) == {"unet": 2e-4}

    shutil.rmtree(tmp, ignore_errors=True)
    print("test_probe OK: stale shards ignored, ranking is mean-rank across objectives")


if __name__ == "__main__":
    main()
