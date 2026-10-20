"""Phone/browser camera, live zone editing, side-by-side frames, network info."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.db.database import Database
from app.main import create_app
from app.services.surveillance_service import default_source_factory
from tests.conftest import FakeDetector, wait_until

SQUARE = [[100, 100], [300, 100], [300, 200], [100, 200]]


def jpeg(w=320, h=240) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def browser_client(settings):
    """App wired with the real source factory, so source_type 'browser' uses BrowserFrameSource."""
    app = create_app(settings, detector_factory=lambda _s: FakeDetector(),
                     source_factory=default_source_factory, database=Database(None))
    with TestClient(app) as client:
        yield client


# -- phone / browser camera ------------------------------------------------------------
def test_camera_socket_is_refused_without_an_active_browser_session(browser_client):
    with browser_client.websocket_connect("/ws/camera") as ws:
        assert ws.receive_json()["type"] == "error"


def test_browser_camera_session_end_to_end(browser_client):
    client = browser_client
    r = client.post("/api/v1/sessions", json={"source_type": "browser"})
    assert r.status_code == 201 and r.json()["source_label"] == "browser:camera"
    sid = r.json()["id"]

    with client.websocket_connect("/ws/camera") as ws:
        assert ws.receive_json()["type"] == "ready"
        for _ in range(12):
            ws.send_bytes(jpeg())
        assert wait_until(lambda: client.get("/api/v1/objects").json()["objects"] != [])
        status = client.app.state.service.status()
        assert status["source_kind"] == "browser" and status["browser_frames"] >= 1 and status["running"]

        full = client.get("/api/v1/frame").content
        plain = client.get("/api/v1/frame?view=plain").content
        assert full[:2] == b"\xff\xd8" and plain[:2] == b"\xff\xd8" and full != plain

        stopped = client.post(f"/api/v1/sessions/{sid}/stop")
        assert stopped.status_code == 200 and stopped.json()["status"] == "stopped"
        seen = []
        for _ in range(20):
            msg = ws.receive_json()
            seen.append(msg["type"])
            if msg["type"] == "stopped":
                break
        assert "stopped" in seen
    assert client.app.state.service.browser_source is None
    assert client.get("/health").json()["session_running"] is False


def test_browser_camera_can_be_disabled(make_client):
    client, _ = make_client(allow_browser_camera=False)
    r = client.post("/api/v1/sessions", json={"source_type": "browser"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "configuration_error"


def test_a_silent_browser_camera_fails_the_session(settings):
    from app.vision.browser_source import BrowserFrameSource
    app = create_app(settings, detector_factory=lambda _s: FakeDetector(),
                     source_factory=lambda spec, _s: BrowserFrameSource(first_frame_timeout_s=0.3),
                     database=Database(None))
    with TestClient(app) as client:
        sid = client.post("/api/v1/sessions", json={"source_type": "browser"}).json()["id"]
        assert wait_until(lambda: client.get(f"/api/v1/sessions/{sid}").json()["status"] == "failed")
        assert "not sending frames" in client.get(f"/api/v1/sessions/{sid}").json()["error"]


def test_phone_page_and_network_info(make_client):
    client, _ = make_client()
    page = client.get("/phone")
    assert page.status_code == 200 and "getUserMedia" in client.get("/static/js/phone.js").text
    off = client.get("/api/v1/network").json()
    assert off["https_enabled"] is False and off["phone_urls"] == [] and "run_demo" in off["hint"]
    client2, _ = make_client(public_https_port=8443)
    on = client2.get("/api/v1/network").json()
    assert on["https_enabled"] is True and all(u.startswith("https://") and u.endswith(":8443/phone") for u in on["phone_urls"])


# -- live zone editing ---------------------------------------------------------------------
def put_zones(client, zones, ref=(320, 240)):
    return client.put("/api/v1/zones", json={"reference_size": list(ref), "zones": zones})


def test_zone_replace_persist_and_reset(make_client, settings):
    client, _ = make_client()
    r = put_zones(client, [{"id": "vault", "name": "Vault", "restricted": True, "polygon": SQUARE}])
    assert r.status_code == 200
    body = client.get("/api/v1/zones").json()
    assert body["reference_size"] == [320.0, 240.0] and body["zones"][0]["name"] == "Vault"
    saved = json.loads(settings.zones_file.read_text())          # survives a restart
    assert saved["zones"][0]["id"] == "vault" and saved["reference_size"] == [320.0, 240.0]
    assert put_zones(client, []).json()["zones"] == []
    reset = client.post("/api/v1/zones/reset").json()
    assert reset["zones"][0]["id"] == "zone-1" and reset["reference_size"] == [1280.0, 720.0]


@pytest.mark.parametrize("zones", [
    [{"id": "a", "name": "A", "polygon": [[0, 0], [10, 10]]}],                          # < 3 points
    [{"id": "a", "name": "A", "polygon": [[0, 0], [10, 0], [10, 9999]]}],               # outside frame
    [{"id": "a", "name": "A", "polygon": [[0, 0], [10, 0], [10, 10]]}] * 2,             # duplicate ids
    [{"id": "bad/../id", "name": "A", "polygon": [[0, 0], [10, 0], [10, 10]]}],         # unsafe id
    [{"id": f"z{i}", "name": "Z", "polygon": [[0, 0], [10, 0], [10, 10]]} for i in range(11)],
    [{"id": "a", "name": "", "polygon": [[0, 0], [10, 0], [10, 10]]}],                  # empty name
    [{"id": "a", "name": "A", "polygon": [[0, 0, 0], [10, 0], [10, 10]]}],              # 3-D point
])
def test_zone_validation(make_client, zones):
    client, _ = make_client()
    r = put_zones(client, zones)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_zone_edit_applies_to_a_running_session(make_client):
    client, _ = make_client(source_kwargs={"frames": None, "delay": 0.01})
    sid = client.post("/api/v1/sessions", json={"source_type": "camera"}).json()["id"]

    def zone_of_first():
        objs = client.get("/api/v1/objects").json()["objects"]
        return objs[0]["zone_id"] if objs else "none-yet"

    assert wait_until(lambda: zone_of_first() != "none-yet")
    put_zones(client, [])                                              # remove every zone, live
    assert wait_until(lambda: zone_of_first() is None)
    put_zones(client, [{"id": "everywhere", "name": "Everywhere", "polygon": [[0, 0], [320, 0], [320, 240], [0, 240]]}])
    assert wait_until(lambda: zone_of_first() == "everywhere")
    client.post(f"/api/v1/sessions/{sid}/stop")


def test_objects_expose_behaviours_and_explanations(make_client):
    client, _ = make_client(source_kwargs={"frames": None, "delay": 0.01})
    sid = client.post("/api/v1/sessions", json={"source_type": "camera"}).json()["id"]
    assert wait_until(lambda: any(o["behaviors"] for o in client.get("/api/v1/objects").json()["objects"]), 10)
    obj = client.get("/api/v1/objects").json()["objects"][0]
    assert obj["explanation"].startswith("Threat score") and set(obj["factors"]) >= {"object_risk", "proximity"}
    assert {"code", "label", "detail"} <= set(obj["behaviors"][0])
    client.post(f"/api/v1/sessions/{sid}/stop")


# -- recording ---------------------------------------------------------------------------
def test_session_is_recorded_listed_replayed_downloaded_and_deleted(make_client):
    client, _ = make_client(source_kwargs={"frames": 40})
    sid = client.post("/api/v1/sessions", json={"source_type": "video", "video_name": "clip.mp4"}).json()["id"]
    assert wait_until(lambda: client.get(f"/api/v1/sessions/{sid}").json()["status"] == "completed")

    session = client.get(f"/api/v1/sessions/{sid}").json()
    assert session["recording_name"] == f"session_{sid}.avi" and session["recording_seconds"] > 0
    items = client.get("/api/v1/recordings").json()
    assert [i["session_id"] for i in items] == [sid] and items[0]["size_bytes"] > 0

    dl = client.get(f"/api/v1/recordings/{sid}/download")
    assert dl.status_code == 200 and dl.content[:4] == b"RIFF" and "attachment" in dl.headers["content-disposition"]

    with client.stream("GET", f"/api/v1/recordings/{sid}/replay") as r:
        body = b"".join(r.iter_bytes())
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace") and body.count(b"\xff\xd8") >= 5

    assert client.delete(f"/api/v1/recordings/{sid}").status_code == 204
    assert client.get("/api/v1/recordings").json() == []
    assert client.get(f"/api/v1/recordings/{sid}/download").status_code == 404
    assert client.get(f"/api/v1/sessions/{sid}").json()["recording_name"] is None


def test_recording_can_be_switched_off_per_session(make_client):
    client, _ = make_client(source_kwargs={"frames": 20})
    sid = client.post("/api/v1/sessions", json={"source_type": "camera", "record": False}).json()["id"]
    assert wait_until(lambda: client.get(f"/api/v1/sessions/{sid}").json()["status"] == "completed")
    assert client.get(f"/api/v1/sessions/{sid}").json()["recording_name"] is None
    assert client.get("/api/v1/recordings").json() == []


def test_unfinished_and_unknown_recordings_are_not_served(make_client):
    client, _ = make_client(source_kwargs={"frames": None, "delay": 0.01})
    sid = client.post("/api/v1/sessions", json={"source_type": "camera"}).json()["id"]
    assert client.get(f"/api/v1/recordings/{sid}/download").status_code == 404   # still recording
    assert client.get("/api/v1/recordings/9999/replay").status_code == 404
    client.post(f"/api/v1/sessions/{sid}/stop")
    assert client.get(f"/api/v1/recordings/{sid}/download").status_code == 200


def test_model_warmup_runs_in_the_background_and_is_reported(make_client):
    client, _ = make_client()
    assert wait_until(lambda: client.get("/health").json()["model_state"] == "ready")
    assert client.app.state.service.status()["model"] == "ready"
