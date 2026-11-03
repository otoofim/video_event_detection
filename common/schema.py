"""Event schema shared by every stage of the pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field

EVENT_TYPES = ["kill", "round_start", "round_end"]
CLASSES = ["background", *EVENT_TYPES]  # frame-level classes; index 0 is "no event"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class Event(BaseModel):
    """A discrete in-game event: ``{timestamp, event_type, confidence, bbox?}``."""

    timestamp: float = Field(description="Seconds from the start of the stream")
    event_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = Field(default=None, description="[x1, y1, x2, y2] in pixels")
    team: str | None = None


class Anomaly(BaseModel):
    """A flagged irregular-play interval from the temporal model."""

    start: float
    end: float
    score: float
    kind: str | None = None


class Annotation(BaseModel):
    """Ground truth for one clip, written next to the video by the generator."""

    video: str
    fps: float
    duration: float
    width: int
    height: int
    events: list[Event]
    anomalies: list[Anomaly] = []
