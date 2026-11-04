"""Cut a long video into clips the way LoL-V2T was made (one clip per detected scene).

The dataset ran PySceneDetect ``ContentDetector(threshold=15, min_scene_len=180)`` and then a
private "useful scene" classifier; that classifier is not published, so every scene is kept.
"""

from __future__ import annotations

from pathlib import Path


def fixed_chunks(
    n_frames: int, fps: float, chunk_s: float, min_frames: int = 1
) -> list[tuple[int, int]]:
    """``[start, end)`` frame ranges of ``chunk_s`` seconds; a short tail joins the previous one."""
    size = max(int(round(chunk_s * fps)), 1)
    out = [(s, min(s + size, n_frames)) for s in range(0, n_frames, size)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_frames:
        out[-2:] = [(out[-2][0], n_frames)]
    return out


def scene_chunks(
    path: str | Path, threshold: float = 15.0, min_scene_len: int = 180
) -> list[tuple[int, int]]:
    """``[start, end)`` frame ranges of PySceneDetect content scenes (needs ``scenedetect``)."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise ImportError("scene chunking needs `pip install scenedetect`") from exc
    video = open_video(str(path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.auto_downscale = True  # the dataset used PySceneDetect's default downscale
    sm.detect_scenes(video)
    scenes = [(_frame(s), _frame(e)) for s, e in sm.get_scene_list()]
    if not scenes:  # no cut found: the whole video is one scene
        scenes = [(0, _frame(video.duration))]
    return scenes


def _frame(tc) -> int:
    """Frame number of a PySceneDetect ``FrameTimecode`` (0.6: ``get_frames``, 0.7+: property)."""
    return int(tc.frame_num) if hasattr(tc, "frame_num") else int(tc.get_frames())
