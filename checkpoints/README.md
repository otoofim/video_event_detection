# Trained checkpoints

PyTorch `state_dict` files (plain tensors, load with `torch.load(..., weights_only=True)`) plus the architecture, model arguments, feature statistics and the confidence threshold used during training. `manifest.json` lists the same information with each model's test metrics.

Load and evaluate them with `visualisation/experiment_report.ipynb`. All models take LoL-V2T features of shape `(N, 4096, T)` and return per-step logits for `background, kill, objective, teamfight`.

Test metrics below use the `min_conf` chosen on the validation set for each model. Models trained with different label windows (`Label offset`) are scored against their own labels, so only compare numbers within a group.

| Checkpoint | Arch | Params | Label offset (s) | Bg downsampling | Acc | AUROC | Step macro F1 | Event F1 | Description |
|---|---|---|---|---|---|---|---|---|---|
| `detector.pt` | vunet | 1,645,764 | +0 | no | 0.790 | 0.555 | 0.268 | 0.123 | Variational U-Net (CE + reconstruction + KL), first trainer, early stopped |
| `detector_unet.pt` | unet | 1,149,252 | +0 | no | 0.701 | 0.563 | 0.263 | 0.133 | Plain U-Net, first trainer, epoch 30 of 100 |
| `detector_unet_wsp.pt` | unet | 1,149,252 | +0 | no | 0.754 | 0.563 | 0.268 | 0.128 | detector_unet after gradient-free weight-space edits |
| `base_c.pt` | unet | 1,149,252 | +0 | no | 0.808 | 0.591 | 0.274 | 0.135 | U-Net, GPU trainer, no extra regularisation |
| `reg2_c.pt` | unet | 1,149,252 | +0 | no | 0.812 | 0.606 | 0.285 | 0.136 | U-Net, dropout 0.2, input dropout 0.3, noise 0.3, wd 0.01 |
| `reg3_c.pt` | unet | 353,444 | +0 | no | 0.810 | 0.598 | 0.279 | 0.134 | U-Net hidden 32, stronger regularisation |
| `reg4_c.pt` | unet | 121,428 | +0 | no | 0.820 | 0.613 | 0.281 | 0.129 | U-Net hidden 16, input dropout 0.5, wd 0.05 |
| `reg5_c.pt` | unet | 1,149,252 | +0 | no | 0.837 | 0.604 | 0.279 | 0.115 | U-Net, input dropout 0.7 |
| `wt075.pt` | unet | 121,428 | +0 | no | 0.695 | 0.609 | 0.284 | 0.158 | reg4 U-Net, class weight power 0.75 |
| `wt100.pt` | unet | 121,428 | +0 | no | 0.275 | 0.589 | 0.176 | 0.155 | reg4 U-Net, class weight power 1.0 |
| `wt125.pt` | unet | 121,428 | +0 | no | 0.051 | 0.591 | 0.069 | 0.155 | reg4 U-Net, class weight power 1.25 |
| `wt100_h32.pt` | unet | 353,444 | +0 | no | 0.068 | 0.603 | 0.079 | 0.135 | reg3 U-Net, class weight power 1.0 |
| `shift_m9.pt` | unet | 121,428 | -9 | no | 0.769 | 0.604 | 0.276 | 0.102 | wt075 setup, label window -9 s |
| `shift_m6.pt` | unet | 121,428 | -6 | no | 0.749 | 0.615 | 0.283 | 0.113 | wt075 setup, label window -6 s |
| `shift_m3.pt` | unet | 121,428 | -3 | no | 0.762 | 0.611 | 0.280 | 0.121 | wt075 setup, label window -3 s |
| `shift0.pt` | unet | 121,428 | +0 | no | 0.738 | 0.617 | 0.289 | 0.156 | wt075 setup, label window 0 s |
| `shift_p3.pt` | unet | 121,428 | +3 | no | 0.723 | 0.633 | 0.281 | 0.145 | wt075 setup, label window +3 s |
| `shift_p6.pt` | unet | 121,428 | +6 | no | 0.752 | 0.654 | 0.282 | 0.136 | wt075 setup, label window +6 s |
| `shift_p9.pt` | unet | 121,428 | +9 | no | 0.713 | 0.657 | 0.276 | 0.136 | wt075 setup, label window +9 s |
| `rs0_control.pt` | unet | 121,428 | +6 | no | 0.772 | 0.656 | 0.284 | 0.133 | +6 s labels, class weights 0.75, AUROC-selected (control) |
| `rs5_balance_crops.pt` | unet | 121,428 | +6 | yes | 0.357 | 0.633 | 0.198 | 0.126 | +6 s labels, background downsampling + event-centred crops |
| `ctx_unet_l3.pt` | unet | 294,100 | +6 | no | 0.733 | 0.678 | 0.287 | 0.145 | U-Net 3 levels (receptive field 99 steps) |
| `ctx_unet_l4.pt` | unet | 983,508 | +6 | no | 0.637 | 0.678 | 0.272 | 0.145 | U-Net 4 levels (219 steps) |
| `ctx_unet_l4_big.pt` | unet | 6,418,212 | +6 | no | 0.754 | 0.676 | 0.290 | 0.141 | U-Net 4 levels, hidden 32, 2 blocks per level (6.4M params) |
| `ctx_dilated.pt` | dilated | 336,580 | +6 | no | 0.728 | 0.675 | 0.289 | 0.128 | Dilated residual convolutions 1..32 (127 steps) |
| `ctx_bigru.pt` | bigru | 387,140 | +6 | no | 0.743 | 0.691 | 0.301 | 0.136 | BiGRU, 2 layers, hidden 64 (whole clip), class weights 0.75 |
| `gru_reg0.pt` | bigru | 387,140 | +6 | yes | 0.570 | 0.692 | 0.269 | 0.137 | BiGRU + background downsampling, dropout 0.2 |
| `gru_reg1.pt` | bigru | 162,852 | +6 | yes | 0.453 | 0.688 | 0.238 | 0.135 | BiGRU hidden 32, more regularisation, background downsampling |
| `gru_reg2.pt` | bigru | 387,140 | +6 | yes | 0.606 | 0.693 | 0.279 | 0.138 | BiGRU, dropout 0.4, lr 5e-4, background downsampling (best on val AUROC) |
| `gru_reg3.pt` | bigru | 144,036 | +6 | yes | 0.360 | 0.653 | 0.207 | 0.127 | BiGRU hidden 32, 1 layer, input dropout 0.7 |
| `gru_reg4.pt` | bigru | 387,140 | +6 | yes | 0.432 | 0.686 | 0.231 | 0.140 | BiGRU, strongest regularisation, lr 3e-4 |

Left out on purpose: `reg1` to `reg6` (early runs whose monitor used a fixed confidence threshold), `rs1` to `rs4` (sweep entries stopped before finishing).
