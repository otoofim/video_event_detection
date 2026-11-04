"""Event schema shared by every stage of the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

EVENT_TYPES = ["kill", "objective", "teamfight"]
CLASSES = ["background", *EVENT_TYPES]  # per-step classes; index 0 is "no event"


class Event(BaseModel):
    """A discrete in-game event: ``{timestamp, event_type, confidence, bbox?}``."""

    timestamp: float = Field(description="Seconds from the start of the clip")
    event_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = Field(
        default=None, description="[x1, y1, x2, y2]; unused for feature-based detection"
    )
