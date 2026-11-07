"""Settings of the LoL-V2T extractor (``tsunehiko/create-lol-v2t``, mmaction2 0.x TSN).

Every default here comes from the dataset's own scripts; see ``docs/feature_extraction.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

# mmaction2 model zoo checkpoints named in create-lol-v2t/dataset/make_dataset.py.
_ZOO = "https://download.openmmlab.com/mmaction/recognition/tsn"
CHECKPOINTS = {
    "rgb": f"{_ZOO}/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/"
    "tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20200804-9e15687e.pth",
    "flow": f"{_ZOO}/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/"
    "tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52.pth",
}


@dataclass(frozen=True)
class ExtractConfig:
    frame_interval: int = 13  # INTERVAL in make_dataset.sh: 0.217 s at 60 fps
    rgb_clip_len: int = 1
    flow_clip_len: int = 5  # 5 stacked (x, y) TVL1 flow frames -> 10 input channels
    width: int = 340  # rawframes are resized to 340 x 256 before flow and features
    height: int = 256
    crop: int = 256  # CenterCrop after Resize(-1, 256)
    flow_bound: float = 20.0  # denseflow clips flow to +-20 px and stores it as 8-bit JPEG
    jpeg_roundtrip: bool = True  # reproduce the JPEG loss of the dataset's rawframes
    jpeg_quality: int = 95
    rgb_mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    rgb_std: tuple[float, float, float] = (58.395, 57.12, 57.375)
    flow_mean: float = 128.0
    flow_std: float = 128.0
    batch_size: int = 32  # steps per network forward pass
    # chunking, PySceneDetect settings from make_dataset.sh / caption/split.py
    scene_threshold: float = 15.0
    min_scene_len: int = 180  # frames
