"""One epoch of detection.train on a tiny on-disk dataset in the LoL-V2T layout (plumbing only)."""

import json
import pickle
import sys

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

SENTENCES = ["welcome back", "he gets the kill", "baron is down", "big team fight"]


def _fake_root(root, rng):
    (root / "annotations").mkdir(parents=True)
    for split, fname in [("training", "training"), ("validation", "validation"), ("testing", "testing")]:
        (root / split).mkdir()
        ann = {}
        for k in range(3):
            cid = f"{split}_{k}"
            t = int(rng.integers(9, 30))
            for mod in ("rgb", "flow"):
                arr = rng.standard_normal((1, t, 8)).astype(np.float32)
                (root / split / f"{cid}_{mod}.pkl").write_bytes(pickle.dumps(arr, protocol=3))
            ann[cid] = {
                "duration": t * 0.217,
                "timestamps": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "sentences": SENTENCES,
            }
        (root / "annotations" / f"{fname}.json").write_text(json.dumps(ann))


@pytest.mark.parametrize("arch", ["vunet", "dilated"])
def test_train_one_epoch_writes_checkpoint_and_report(tmp_path, arch):
    from detection.train import main

    _fake_root(tmp_path / "data", np.random.default_rng(0))
    cfg = yaml.safe_load(open("detection/configs/default.yaml"))
    cfg.update(root=str(tmp_path / "data"), epochs=1, norm_clips=3, mc_samples=2, batch_size=2, num_workers=0)
    cfg["model"]["vunet"] = {"hidden": 8, "latent": 4, "levels": 2, "dropout": 0.0}
    cfg["model"]["dilated"] = {"hidden": 8, "dropout": 0.0}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    out, report = tmp_path / "det.pt", tmp_path / "results.json"
    argv = sys.argv
    sys.argv = ["x", "--config", str(tmp_path / "cfg.yaml"), "--out", str(out),
                "--report", str(report), "--arch", arch]
    try:
        main()
    finally:
        sys.argv = argv
    rep = json.loads(report.read_text())
    ckpt = torch.load(out, weights_only=True)
    assert rep["arch"] == ckpt["arch"] == arch and ckpt["in_dim"] == 16
    assert ("uncertainty" in rep["test"]) == (arch == "vunet")
