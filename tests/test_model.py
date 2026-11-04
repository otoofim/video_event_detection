"""Variational U-Net: shapes, padding and masking, loss terms, uncertainty, ablation variant."""

import pytest

torch = pytest.importorskip("torch")

from detection.model import (  # noqa: E402
    VariationalUNet,
    beta_at,
    latent_mask,
    mc_probs,
    vae_loss,
)
from detection.train import IGNORE  # noqa: E402

C = 4


def net(**kw):
    torch.manual_seed(0)
    return VariationalUNet(16, C, hidden=8, latent=4, levels=2, dropout=0.0, **kw)


@pytest.mark.parametrize("t", [1, 30, 32, 37])
def test_shapes_for_any_length(t):
    m = net().eval()
    out = m(torch.randn(2, 16, t))
    assert out["logits"].shape == (2, C, t)
    assert out["recon"].shape == (2, 16, t) and out["target"].shape == (2, 16, t)
    tz = -(-t // 4)
    assert out["mu"].shape == (2, 4, tz) and out["logvar"].shape == (2, 4, tz)


def test_eval_is_deterministic_and_noise_changes_logits():
    m = net().eval()
    x = torch.randn(1, 16, 20)
    a, b = m(x)["logits"], m(x)["logits"]
    assert torch.equal(a, b)
    noisy = m(x, noise=torch.randn(1, 4, 5))["logits"]
    assert not torch.allclose(a, noisy)


def test_reconstruction_uses_only_the_latent():
    m = net().eval()
    out = m(torch.randn(1, 16, 20))
    assert torch.allclose(out["recon"], m.reconstruct(out["mu"], 20))


def test_latent_mask_keeps_any_window_with_a_valid_step():
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert latent_mask(mask, 4).tolist() == [[True, True, False]]


def test_padding_is_masked_out_of_every_loss_term():
    m = net().eval()
    y = torch.zeros(1, 16, dtype=torch.long)
    y[:, 5:] = IGNORE  # steps 5..15 are padding; latent steps 2, 3 (steps 8..15) fully padded
    out = m(torch.randn(1, 16, 16))
    total, parts = vae_loss(out, y, None, 1.0, 1.0)
    bad = {k: v.clone() if torch.is_tensor(v) else v for k, v in out.items()}
    for k in ("logits", "recon", "target"):
        bad[k][..., 5:] = 1e3
    for k in ("mu", "logvar"):
        bad[k][..., 2:] = 5.0
    total2, parts2 = vae_loss(bad, y, None, 1.0, 1.0)
    assert set(parts) == {"ce", "recon", "kl"}
    for k in parts:
        assert parts[k] == pytest.approx(parts2[k], rel=1e-6), k
    assert total.item() == pytest.approx(total2.item(), rel=1e-6)


def test_kl_is_zero_for_a_standard_normal_posterior():
    m = net().eval()
    x = torch.randn(1, 16, 8)
    out = m(x)
    out["mu"], out["logvar"] = torch.zeros_like(out["mu"]), torch.zeros_like(out["logvar"])
    _, parts = vae_loss(out, torch.zeros(1, 8, dtype=torch.long), None, 1.0, 1.0)
    assert parts["kl"] == pytest.approx(0.0, abs=1e-7)


def test_beta_ramps_linearly_then_holds():
    assert beta_at(0, 100, 0.5) == 0.0
    assert beta_at(50, 100, 0.5) == pytest.approx(0.25)
    assert beta_at(500, 100, 0.5) == 0.5
    assert beta_at(10, 0, 0.5) == 0.5


def test_mc_probs_mean_and_spread():
    m = net().eval()
    mean, std = mc_probs(m, torch.randn(1, 16, 12), n_samples=8)
    assert mean.shape == std.shape == (1, C, 12)
    assert torch.allclose(mean.sum(1), torch.ones(1, 12), atol=1e-5)
    assert (std > 0).any()
    _, std1 = mc_probs(m, torch.randn(1, 16, 12), n_samples=1)
    assert torch.count_nonzero(std1) == 0


def test_plain_unet_ablation_has_no_variational_terms():
    m = net(variational=False).eval()
    out = m(torch.randn(1, 16, 10))
    assert out["mu"] is None and out["recon"] is None
    _, parts = vae_loss(out, torch.zeros(1, 10, dtype=torch.long), None, 1.0, 1.0)
    assert set(parts) == {"ce"}


@pytest.mark.parametrize("arch", ["vunet", "unet", "dilated"])
def test_training_loss_backpropagates_for_every_arch(arch):
    from detection.model import build_model
    from detection.train import collate, compute_loss

    torch.manual_seed(0)
    m = build_model(arch, 16, hidden=8, latent=4, levels=2, dropout=0.0).train()
    x, y = collate(
        [
            (torch.randn(9, 16), torch.randint(0, C, (9,))),
            (torch.randn(5, 16), torch.randint(0, C, (5,))),
        ]
    )
    loss, parts = compute_loss(m, x, y, None, lam=1.0, beta=0.1)
    loss.backward()
    assert torch.isfinite(loss) and "ce" in parts
    assert ("kl" in parts) == (arch == "vunet")


def test_detector_loads_pt_and_onnx_and_they_agree(model_dir):
    import numpy as np

    from detection.infer import Detector

    feats = np.random.default_rng(1).standard_normal((23, 16)).astype(np.float32)
    pt, onnx = Detector(model_dir / "detector.pt"), Detector(model_dir / "detector.onnx")
    assert np.allclose(pt.probs(feats), onnx.probs(feats), atol=1e-4)
    events, probs = pt.detect(feats, duration=5.0)
    assert probs.shape == (23, C)
    mean, std = pt.sample(feats, n_samples=4)
    assert mean.shape == std.shape == (23, C) and (std >= 0).all()
