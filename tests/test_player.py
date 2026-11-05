"""The /player page is served by the app (browser behaviour is checked manually, see README)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from serving.app import create_app  # noqa: E402


def test_player_page_is_served_without_an_extractor(model_dir):
    with TestClient(create_app(str(model_dir), extractor=None)) as c:
        r = c.get("/player")
        assert c.get("/").status_code == 200  # / redirects to the player
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for needle in ('id="video"', 'id="timeline"', 'id="events"', "/detect_video", "/meta"):
        assert needle in r.text
