"""End-to-end API test with a public-domain face (NASA astronaut portrait, via scikit-image's repo).
Uses the real models and real Rekognition/S3, so it is skipped without AWS credentials.
Cleans up the person it creates (DB row, Rekognition face, S3 objects)."""
import os
import ssl
import urllib.request
from pathlib import Path

import boto3
import certifi
import cv2
import pytest

CACHE = Path(__file__).parent / ".cache"
URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png"


def _creds() -> bool:
    try:
        boto3.client("sts", region_name="us-east-1").get_caller_identity()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _creds(), reason="needs AWS credentials (Rekognition + S3)")


@pytest.fixture(scope="module")
def face_jpeg() -> bytes:
    CACHE.mkdir(exist_ok=True)
    png = CACHE / "astronaut.png"
    if not png.exists():
        ctx = ssl.create_default_context(cafile=certifi.where())
        png.write_bytes(urllib.request.urlopen(URL, timeout=30, context=ctx).read())
    return cv2.imencode(".jpg", cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


TEST_COLLECTION = "hid-liveness-faces-pytest"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Isolated from the running app: temp data dir + its own Rekognition collection (deleted after)."""
    os.environ.pop("BENCH_TOKEN", None)
    os.environ["HID_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    os.environ["HID_COLLECTION"] = TEST_COLLECTION
    rek = boto3.client("rekognition", region_name="us-east-1")
    try:
        rek.create_collection(CollectionId=TEST_COLLECTION)
    except rek.exceptions.ResourceAlreadyExistsException:
        pass
    from fastapi.testclient import TestClient

    from backend import main
    try:
        with TestClient(main.asgi) as c:
            yield c, main
    finally:
        rek.delete_collection(CollectionId=TEST_COLLECTION)


def test_enroll_verify_open_set_and_delete(client, face_jpeg):
    c, main = client
    pid = "pytest-astro"
    c.delete(f"/api/people/{pid}")
    r = c.post("/api/enroll", data={"name": pid}, files=[("files", ("a.jpg", face_jpeg, "image/jpeg"))])
    assert r.status_code == 200, r.text
    try:
        frames = [("files", (f"f{i}.jpg", face_jpeg, "image/jpeg")) for i in range(3)]
        v = c.post("/api/verify", data={"person_id": pid, "variant": "A"}, files=frames).json()
        assert v["decision"] == "ALLOW", v["reason"]
        assert v["raw"]["similarity"] > 99
        assert set(v["timings"]) >= {"decode", "detect", "quality", "liveness", "cues", "match", "decision", "total"}
        o = c.post("/api/verify", data={"mode": "open"}, files=frames).json()
        assert o["raw"]["person_id"] == pid
        dark = cv2.imencode(".jpg", (cv2.imdecode(__import__("numpy").frombuffer(face_jpeg, "uint8"), 1) * 0.2).astype("uint8"))[1].tobytes()
        d = c.post("/api/verify", data={"person_id": pid}, files=[("files", ("d.jpg", dark, "image/jpeg"))]).json()
        assert d["decision"] == "REQUEST_VERIFICATION" and "too dark" in d["reason"]
        assert c.post("/api/verify", data={"person_id": "nobody-here"}, files=frames).status_code == 404
    finally:
        assert c.delete(f"/api/people/{pid}").status_code == 200
    assert pid not in c.get("/api/users").json()


def test_token_auth(face_jpeg):
    from fastapi.testclient import TestClient

    from backend.main import TokenAuth, app
    c = TestClient(TokenAuth(app, "s3cret"))
    assert c.get("/api/config").status_code == 401
    assert c.get("/api/config", headers={"x-bench-token": "s3cret"}).status_code == 200
    r = c.get("/bench?token=s3cret", follow_redirects=False)
    assert r.status_code == 303 and "bench_token=s3cret" in r.headers["set-cookie"] and r.headers["location"] == "/bench"
    assert c.get("/healthz").status_code == 200
