import cv2
import numpy as np
import pytest

from tests.conftest import FakeDetector, FakeSource, person_walk, wait_until

ENDED = {"completed", "stopped", "failed"}


def start(client, **body):
    return client.post("/api/v1/sessions", json=body or {"source_type": "camera"})


def wait_ended(client, session_id, timeout=15.0):
    return wait_until(lambda: client.get(f"/api/v1/sessions/{session_id}").json()["status"] in ENDED, timeout)


# -- basics ------------------------------------------------------------------------------
def test_health_and_dashboard(make_client):
    client, _ = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["session_running"] is False
    page = client.get("/")
    assert page.status_code == 200 and "VisionShield AI" in page.text
    assert client.get("/static/js/dashboard.js").status_code == 200


def test_config_is_safe_and_describes_scoring(make_client):
    client, _ = make_client()
    body = client.get("/api/v1/config").json()
    assert body["stream_configured"] is True
    assert "secret" not in client.get("/api/v1/config").text and "rtsp" not in client.get("/api/v1/config").text
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-9
    assert body["levels"] == {"LOW": "0-30", "MEDIUM": "31-60", "HIGH": "61-100"}


def test_zones_and_sources(make_client):
    client, _ = make_client()
    zones = client.get("/api/v1/zones").json()
    assert zones["reference_size"] == [1280.0, 720.0] and zones["zones"][0]["restricted"] is True
    sources = client.get("/api/v1/sources").json()
    assert sources["videos"] == ["clip.mp4"] and "rtsp" not in str(sources)


def test_unknown_routes_use_consistent_error_shape(make_client):
    client, _ = make_client()
    r = client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_404"


# -- request validation -----------------------------------------------------------------
@pytest.mark.parametrize("body", [
    {"source_type": "ftp"},
    {"source_type": "video"},
    {"source_type": "camera", "video_name": "clip.mp4"},
    {"source_type": "video", "video_name": "../secret.mp4"},
    {"source_type": "video", "video_name": "C:\\Windows\\x.mp4"},
    {"source_type": "camera", "url": "rtsp://evil/stream"},
])
def test_session_requests_are_validated(make_client, body):
    client, _ = make_client()
    r = client.post("/api/v1/sessions", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_event_query_validation(make_client):
    client, _ = make_client()
    assert client.get("/api/v1/events?level=SEVERE").status_code == 422
    assert client.get("/api/v1/events?page=0").status_code == 422
    assert client.get("/api/v1/events?page_size=1000").status_code == 422
    assert client.get("/api/v1/events?start=2025-02-01T00:00:00&end=2025-01-01T00:00:00").status_code == 422
    assert client.get("/api/v1/events/12345").status_code == 404
    assert client.get("/api/v1/sessions/999").status_code == 404
    assert client.post("/api/v1/sessions/999/stop").status_code == 404


def test_missing_video_and_unconfigured_stream(make_client):
    client, _ = make_client()
    r = client.post("/api/v1/sessions", json={"source_type": "video", "video_name": "missing.mp4"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "source_not_found"
    client2, _ = make_client(stream_url=None)
    r = client2.post("/api/v1/sessions", json={"source_type": "stream"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "configuration_error"


def test_unavailable_camera_returns_503_and_leaves_no_running_session(make_client):
    client, _ = make_client(source_kwargs={"fail_open": True})
    r = start(client, source_type="camera")
    assert r.status_code == 503 and r.json()["error"]["code"] == "source_unavailable"
    assert client.get("/health").json()["session_running"] is False
    assert FakeSource.instances[-1].closed is True


# -- full session flow ------------------------------------------------------------------
def test_recorded_video_session_end_to_end(make_client):
    client, detector = make_client(source_kwargs={"frames": 60})
    r = start(client, source_type="video", video_name="clip.mp4")
    assert r.status_code == 201
    session = r.json()
    assert session["source_label"] == "video:clip.mp4" and session["status"] == "running"
    assert wait_ended(client, session["id"])

    final = client.get(f"/api/v1/sessions/{session['id']}").json()
    assert final["status"] == "completed" and final["frames_processed"] == 60 and final["error"] is None
    assert FakeSource.instances[-1].closed and detector.warmed

    listing = client.get("/api/v1/sessions").json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == session["id"]

    events = client.get("/api/v1/events", params={"session_id": session["id"]}).json()
    assert events["total"] >= 2
    types = {e["event_type"] for e in events["items"]}
    assert {"zone_entry", "threat_level"} <= types
    high = client.get("/api/v1/events?level=HIGH").json()
    assert high["total"] >= 1 and all(e["level"] == "HIGH" for e in high["items"])
    assert client.get("/api/v1/events?page=1&page_size=1").json()["items"].__len__() == 1

    detail = client.get(f"/api/v1/events/{high['items'][0]['id']}").json()
    assert set(detail["factors"]) == {"object_risk", "confidence", "movement", "proximity", "persistence"}
    assert "Threat score" in detail["explanation"] and detail["observation"]["track_id"] == 1
    assert detail["timestamp"].endswith("Z")

    summary = client.get("/api/v1/analytics/summary").json()
    assert summary["total"] == events["total"] and summary["by_level"]["HIGH"] >= 1
    assert summary["by_class"]["person"] == summary["total"]
    future = client.get("/api/v1/analytics/summary?start=2999-01-01T00:00:00Z").json()
    assert future["total"] == 0

    # frame snapshot of the last annotated frame (the final placeholder after the video ended)
    assert client.get("/api/v1/frame").headers["content-type"] == "image/jpeg"


def test_stop_releases_camera_worker_and_prevents_duplicates(make_client):
    client, detector = make_client(source_kwargs={"frames": None, "delay": 0.01})
    session = start(client, source_type="camera").json()
    service = client.app.state.service
    assert wait_until(lambda: client.get("/api/v1/objects").json()["objects"] != [])
    assert service.is_running

    dup = start(client, source_type="camera")           # second capture loop must be refused
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "session_conflict"
    assert len(FakeSource.instances) == 1

    objs = client.get("/api/v1/objects").json()
    assert objs["active"] is True and objs["objects"][0]["track_id"] == 1
    assert objs["objects"][0]["level"] in {"LOW", "MEDIUM", "HIGH"}

    stopped = client.post(f"/api/v1/sessions/{session['id']}/stop")
    assert stopped.status_code == 200 and stopped.json()["status"] == "stopped"
    assert FakeSource.instances[0].closed is True        # camera released
    assert not service.is_running and service._worker is None
    assert client.get("/api/v1/objects").json() == {"session_id": session["id"], "active": False, "objects": []}

    again = client.post(f"/api/v1/sessions/{session['id']}/stop")   # idempotent
    assert again.status_code == 200 and again.json()["status"] == "stopped"

    second = start(client, source_type="camera")        # a new session can start afterwards
    assert second.status_code == 201
    client.post(f"/api/v1/sessions/{second.json()['id']}/stop")
    assert detector.resets == 2 and not service.is_running


def test_source_failure_mid_session_is_recorded(make_client):
    class DyingSource(FakeSource):
        def read(self):
            if self.read_count >= 5:
                from app.core.exceptions import SourceUnavailableError
                raise SourceUnavailableError("lost connection to camera:0")
            return super().read()

    client, _ = make_client()
    client.app.state.service._source_factory = lambda spec, _s: DyingSource(spec, frames=None)
    sid = start(client, source_type="camera").json()["id"]
    assert wait_ended(client, sid)
    row = client.get(f"/api/v1/sessions/{sid}").json()
    assert row["status"] == "failed" and "lost connection" in row["error"]
    assert client.app.state.service.status()["state"] == "failed"


def test_shutdown_stops_running_worker(make_client):
    client, detector = make_client(source_kwargs={"frames": None, "delay": 0.01})
    start(client, source_type="camera")
    service = client.app.state.service
    service.shutdown()
    assert not service.is_running and FakeSource.instances[0].closed and detector.closed


# -- websocket ---------------------------------------------------------------------------
def test_websocket_sends_status_then_live_events(make_client):
    client, _ = make_client(source_kwargs={"frames": None, "delay": 0.02})
    with client.websocket_connect("/ws/events") as ws:
        first = ws.receive_json()
        assert first["type"] == "status" and first["running"] is False
        sid = start(client, source_type="camera").json()["id"]
        seen = []
        for _ in range(200):
            msg = ws.receive_json()
            seen.append(msg["type"])
            if msg["type"] == "event":
                assert msg["event"]["level"] in {"MEDIUM", "HIGH", "LOW"} and msg["event"]["session_id"] == sid
                break
        else:
            pytest.fail(f"no event received; saw {set(seen)}")
        client.post(f"/api/v1/sessions/{sid}/stop")
    assert client.app.state.hub.client_count == 0   # subscriber cleaned up on disconnect


def test_final_status_message_says_not_running(make_client):
    """Regression: the dashboard kept Start disabled because the last status said running=true."""
    client, _ = make_client(source_kwargs={"frames": 20, "delay": 0.01})
    with client.websocket_connect("/ws/events") as ws:
        ws.receive_json()
        start(client, source_type="camera")
        for _ in range(500):
            msg = ws.receive_json()
            if msg["type"] == "status" and msg["state"] == "completed":
                assert msg["running"] is False
                break
        else:
            pytest.fail("never saw the completed status")


# -- image analysis ----------------------------------------------------------------------
def png_bytes(w=320, h=240) -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_image_analysis(make_client):
    client, detector = make_client(detector=FakeDetector(lambda fid: [person_walk(30)]))
    r = client.post("/api/v1/images/analyze", files={"file": ("shot.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["object_count"] == 1 and body["width"] == 320 and body["annotated_image"].startswith("data:image/jpeg;base64,")
    obj = body["objects"][0]
    assert obj["class_name"] == "person" and obj["factors"]["movement"]["known"] is False
    assert "temporal" in body["note"]


def test_image_upload_validation(make_client):
    client, _ = make_client(max_image_upload_mb=1)
    bad_ext = client.post("/api/v1/images/analyze", files={"file": ("x.exe", b"abc", "application/octet-stream")})
    assert bad_ext.status_code == 415
    corrupt = client.post("/api/v1/images/analyze", files={"file": ("x.png", b"not an image", "image/png")})
    assert corrupt.status_code == 422 and corrupt.json()["error"]["code"] == "invalid_image"
    empty = client.post("/api/v1/images/analyze", files={"file": ("x.png", b"", "image/png")})
    assert empty.status_code == 422
    huge = client.post("/api/v1/images/analyze", files={"file": ("x.png", b"0" * (1024 * 1024 + 10), "image/png")})
    assert huge.status_code == 413


def test_video_upload_rejects_bad_types_and_unreadable_files(make_client, video_dir):
    client, _ = make_client()
    assert client.post("/api/v1/videos/upload", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    before = set(p.name for p in video_dir.iterdir())
    r = client.post("/api/v1/videos/upload", files={"file": ("a.mp4", b"definitely not video", "video/mp4")})
    assert r.status_code == 422
    assert set(p.name for p in video_dir.iterdir()) == before   # rejected file was removed
