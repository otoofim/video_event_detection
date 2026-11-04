# How LoL-V2T features were made (Step 0 of `docs/plan.md`)

Source read: [`tsunehiko/create-lol-v2t`](https://github.com/tsunehiko/create-lol-v2t) at commit
`d4a7849` (2021-02-16), mainly `dataset/make_dataset.sh`, `dataset/make_dataset.py`,
`dataset/feature_extraction.py`, `caption/split.py`, `caption/make_caption.py`, `caption/choose.py`
and `tools/validation.py`.

The repo imports its extractor from an mmaction2 fork
(`mmaction2/tools/data/lol/tsn_feature_extraction_custom.py`,
`mmaction2/tools/data/build_rawframes_custom.py`) that is **not published**. Where this page
describes that step it follows the public script it was evidently adapted from, mmaction2 0.x
`tools/data/activitynet/tsn_feature_extraction.py`. Each fact below is marked
**[source]** (read in the repo) or **[inferred]** (from mmaction2 0.x or from arithmetic).

## Settled facts

| Question | Answer | |
|---|---|---|
| Frame interval | `INTERVAL=13` frames per step | [source] `make_dataset.sh` |
| Frame size before the networks | rawframes resized to 340 x 256 (`--new-width 340 --new-height 256`) | [source] |
| Optical flow | TVL1 (`METHOD="tvl1"`, denseflow via `build_rawframes`), computed on the resized frames | [source] |
| Frames per step | RGB `clip_len=1`; Flow `clip_len=5` (5 x/y flow pairs stacked = 10 channels) | [source] `clip_lens` in `make_dataset.py` |
| Checkpoints | `tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20200804-9e15687e` and `tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52` | [source] |
| Layer read | ResNet-50 backbone output, global-average-pooled to 2048 (the shipped RGB file is `(1, 109, 2048)`) | [inferred] |
| Sampling | `UntrimmedSampleFrames`: centres at `13 // 2 + 13 k`, `clip_len` frames around each centre | [inferred] |
| Normalisation | RGB mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`, RGB order; flow mean 128, std 128 | [inferred] |
| Crop | `Resize(-1, 256)` (a no-op on 340 x 256) then `CenterCrop(256)` | [inferred] |
| Flow encoding | denseflow writes flow as 8-bit JPEG, clipped to +-20 px: `(f + 20) / 40 * 255` | [inferred] |

### Correction to the paper

The paper says "Kinetics-400 without fine-tuning". The checkpoints the code loads are the
mmaction2 **ActivityNet** TSN models: ResNet-50 initialised from Kinetics-400 and then trained on
ActivityNet v1.3 (200 classes). The features are therefore ActivityNet-tuned. `preprocess/` uses
the checkpoints the code names.

### 109 versus 91 steps

The 0.26 s in the paper is 13 frames at 50 fps. With `UntrimmedSampleFrames`, a clip of `N` frames
gives `ceil((N - 6) / 13)` steps. The inspected 23.6 s clip:

| Assumed fps | Frames | Steps |
|---|---|---|
| 50 | 1180 | 91 (what the paper's 0.26 s predicts) |
| **60** | **1416** | **109 (what the file has)** |

So the source videos are most likely 60 fps and one step is 13/60 = 0.217 s. **[inferred]**: it
matches exactly, but no source video was checked. The preprocessing module keeps the interval
in frames (13), as the dataset did, so the step length follows the video's frame rate, and it
reports each step's exact time.

Consequence for labels: step `k` is centred at `(6 + 13k) / fps`; `common.lolv2t.step_times`
assumes `(k + 0.5) * duration / T`. For the 109-step clip the two differ by under 0.01 s, so the
uniform assumption used by the labels is fine.

## Chunking: how long videos became clips

1. **Scene detection** **[source]** `caption/split.py`: PySceneDetect `ContentDetector(threshold=15,
   min_scene_len=180)` (`PYSCENEDETECT_THRESHOLD=15` in `make_dataset.sh`; 180 frames = 3 s at 60
   fps), using PySceneDetect's default downscale. Each scene is cut out with ffmpeg and
   re-encoded (`libx264 -preset fast -crf 21`). A clip is one scene.
2. **"Useful" filter** **[source]** `caption/choose.py`: a ResNeXt-50 binary classifier, run on the
   middle frame of each scene with threshold 0.001, drops non-gameplay scenes (replays, desk,
   crowd). Its weights (`best_cpu.pkl`) are **not published**, so this step cannot be reproduced.
   `preprocess/` keeps every scene and says so in its output.
3. **Caption filter**: scenes with no overlapping commentary are dropped. This does not apply
   to new videos.

There is no fixed chunk length: clip lengths come from the scene cuts.

## Still open

- The fork's exact code: whether it used `feature_extraction=True` pooling or another layer,
  and how flow indices near the end of a clip are clipped (flow has `N - 1` frames). The
  acceptance script (`python -m preprocess.verify`) is how to settle this.
- Video availability: the YouTube sources may have been removed or re-encoded since 2020.
- The source frame rate is inferred, not observed.
