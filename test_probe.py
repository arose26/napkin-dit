"""Two ways the probe reducer can silently corrupt a result, both seen for real:

1. Reading the wrong shards. If it globbed the directory instead of the requested LR grid, a
   leftover shard from an earlier --lrs set would win on stale numbers and get baked into
   out/lr.json for every subsequent run.
2. Publishing from a shard. `probe --backbone unet` reduces over unet alone; if that writes
   the global lr.json, the next shard overwrites it with the other backbone alone. Observed
   live: lr.json held {"unet": 5e-4} with no dit entry while the dit shards were running.

train_one is stubbed out, so this is pure bookkeeping -- no GPU, runs in a second.
    python3 test_probe.py
"""
import json, pathlib, shutil, tempfile, types
import napkin_dit as N


def main():
    tmp = pathlib.Path(tempfile.mkdtemp())
    N.OUT = tmp
    N.train_one = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("trained a shard that already existed"))

    def shard(name, eps, flow):
        (tmp / "probe").mkdir(exist_ok=True)
        (tmp / "probe" / f"{name}.json").write_text(json.dumps({"eps": eps, "flow": flow}))

    def reset():
        shutil.rmtree(tmp / "probe", ignore_errors=True)
        (tmp / "lr.json").unlink(missing_ok=True)
        (tmp / "probe").mkdir()

    lrs = [1e-4, 2e-4]
    def ns(backbone, lrs_, allow=False):
        return types.SimpleNamespace(backbone=backbone, objective=["eps", "flow"],
                                     lrs=lrs_, bs=128, steps=8, allow_boundary_lr=allow)
    # allow=True on these: they exercise shard READING, and their toy 2-point grids
    # happen to peak at the top, which the boundary guard would otherwise refuse.
    both, one = ns(list(N.BACKBONES), lrs, allow=True), ns(["unet"], lrs, allow=True)

    # (2) a sharded invocation must NOT publish, even with its own grid complete
    reset(); shard("unet-0.0001", 0.9, 0.9); shard("unet-0.0002", 0.2, 0.3)
    N.cmd_probe(one)
    assert not (tmp / "lr.json").exists(), "a single-backbone shard published lr.json"

    # (1) a leftover shard from a wider grid, with the best losses on the board by a mile,
    #     must not leak into the ranking of the 2-LR grid actually asked for
    shard("unet-0.001", 0.01, 0.01)
    shard("dit-0.0001", 0.9, 0.9); shard("dit-0.0002", 0.4, 0.4)
    N.cmd_probe(both)
    best = json.loads((tmp / "lr.json").read_text())
    assert best == {"unet": 2e-4, "dit": 2e-4}, f"stale shard leaked in: picked {best}"

    # (3) a pick at the TOP of the grid is not a pick, it is a grid that was too narrow.
    #     This is the real failure that nearly shipped: both backbones chose 5e-4 out of
    #     {1e-4,2e-4,5e-4} with the DiT still improving 34% per grid step, i.e. its optimum
    #     was outside the grid and the UNet's was not -- unequal under-tuning, which would
    #     have manufactured the pre-registered "UNet wins" result.
    reset()
    shard("unet-0.0001", 0.9, 0.9); shard("unet-0.0002", 0.5, 0.5)
    shard("dit-0.0001", 0.9, 0.9); shard("dit-0.0002", 0.5, 0.5)
    N.cmd_probe(ns(list(N.BACKBONES), lrs))            # best is 2e-4 == max(grid)
    assert not (tmp / "lr.json").exists(), "published a boundary LR without being asked to"
    N.cmd_probe(both)
    assert json.loads((tmp / "lr.json").read_text()) == {"unet": 2e-4, "dit": 2e-4}, \
        "--allow-boundary-lr did not let a deliberate boundary pick through"

    # (3b) a ONE-POINT grid has no boundary to be at. Every sharded call passes a single
    #      --lrs, so without this the shards emit a refusal every time and the log fills with
    #      false alarms about a condition that cannot mean anything for a grid of size 1.
    reset()
    shard("unet-0.0001", 0.5, 0.5); shard("dit-0.0001", 0.5, 0.5)
    N.cmd_probe(ns(list(N.BACKBONES), [1e-4]))          # NOT allow=True -- must not refuse
    assert (tmp / "lr.json").exists(), "a one-point grid was refused as a boundary pick"
    assert json.loads((tmp / "lr.json").read_text()) == {"unet": 1e-4, "dit": 1e-4}

    # (4) a diverged shard (nan loss) must rank LAST, never win
    reset()
    for b in ("unet", "dit"):
        shard(f"{b}-0.0001", 0.5, 0.5); shard(f"{b}-0.0002", float("nan"), float("nan"))
    N.cmd_probe(both)
    assert json.loads((tmp / "lr.json").read_text()) == {"unet": 1e-4, "dit": 1e-4}, \
        "a diverged (nan) run won the ranking"

    # NOTE there is deliberately no "missing backbone blocks the publish" case: cmd_probe
    # TRAINS a missing shard before reducing, so that state is only reachable through a
    # sharded call, which case (2) already covers. A test for it would be asserting a
    # behaviour this function does not have.

    shutil.rmtree(tmp, ignore_errors=True)
    print("test_probe OK: shards do not publish, stale grids ignored, "
          "boundary picks refused, nan ranks last")


if __name__ == "__main__":
    main()
