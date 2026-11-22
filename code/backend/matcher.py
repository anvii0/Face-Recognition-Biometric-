"""Identity match via Amazon Rekognition CompareFaces (us-east-1 only)."""
from __future__ import annotations

import threading
import time
from typing import Callable

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


class RekognitionMatcher:
    def __init__(self, region: str = "us-east-1", collection: str = "hid-liveness-faces") -> None:
        if region != "us-east-1":
            raise ValueError("only us-east-1 is allowed in this sandbox")
        self.collection = collection
        self.client = boto3.client("rekognition", region_name=region, config=Config(
            connect_timeout=2, read_timeout=4, retries={"max_attempts": 2, "mode": "standard"}))

    def keep_warm(self, reference: "Callable[[], bytes | None]", interval_s: float = 120.0) -> threading.Event:
        """Measured: even on a warm TLS connection the first *real* CompareFaces took ~2.07 s
        (later calls 0.35-0.6 s). So warm with a real CompareFaces(ref, ref) at startup, every
        interval_s, and whenever the returned Event is set (e.g. right after enrollment)."""
        kick = threading.Event()

        def loop() -> None:
            while True:
                ref = reference()
                try:
                    if ref is not None:
                        self.client.compare_faces(SourceImage={"Bytes": ref}, TargetImage={"Bytes": ref})
                    else:  # no enrolled face yet: at least keep the TLS connection open
                        self.client.list_collections(MaxResults=1)
                except (ClientError, BotoCoreError):
                    pass
                kick.wait(interval_s)
                kick.clear()
        threading.Thread(target=loop, daemon=True, name="rekognition-warm").start()
        return kick

    def compare(self, reference_jpeg: bytes, probe_jpeg: bytes) -> tuple[float | None, str | None, float]:
        """Returns (best similarity 0-100 | None, error | None, elapsed ms).
        SimilarityThreshold=0 so a score is returned even for non-matches (needed for FAR sweeps)."""
        t0 = time.perf_counter()
        try:
            r = self.client.compare_faces(SourceImage={"Bytes": reference_jpeg}, TargetImage={"Bytes": probe_jpeg},
                                          SimilarityThreshold=0, QualityFilter="NONE")
            sims = [m["Similarity"] for m in r.get("FaceMatches", [])]
            sim = max(sims) if sims else 0.0
            return float(sim), None, (time.perf_counter() - t0) * 1e3
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = {"InvalidParameterException": "no face found by matcher",
                   "ExpiredTokenException": "AWS credentials expired",
                   "ExpiredToken": "AWS credentials expired"}.get(code, f"Rekognition {code}")
            return None, msg, (time.perf_counter() - t0) * 1e3
        except BotoCoreError as e:
            return None, f"Rekognition unreachable ({type(e).__name__})", (time.perf_counter() - t0) * 1e3

    # ---------- collection (1:N search, duplicate check, enrollment index) ----------
    def search(self, probe_jpeg: bytes, max_faces: int = 3) -> tuple[list[tuple[str, float]], str | None, float]:
        """Returns ([(person_id, similarity)], error, ms). Threshold 0 so near-misses are visible."""
        t0 = time.perf_counter()
        try:
            r = self.client.search_faces_by_image(CollectionId=self.collection, Image={"Bytes": probe_jpeg},
                                                  MaxFaces=max_faces, FaceMatchThreshold=0, QualityFilter="NONE")
            hits = [(m["Face"]["ExternalImageId"], float(m["Similarity"])) for m in r.get("FaceMatches", [])]
            return hits, None, (time.perf_counter() - t0) * 1e3
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = "no face found by matcher" if code == "InvalidParameterException" else f"Rekognition {code}"
            return [], msg, (time.perf_counter() - t0) * 1e3
        except BotoCoreError as e:
            return [], f"Rekognition unreachable ({type(e).__name__})", (time.perf_counter() - t0) * 1e3

    def index(self, person_id: str, jpeg: bytes) -> str:
        r = self.client.index_faces(CollectionId=self.collection, Image={"Bytes": jpeg}, ExternalImageId=person_id,
                                    MaxFaces=1, QualityFilter="NONE", DetectionAttributes=[])
        if not r["FaceRecords"]:
            raise ValueError("Rekognition found no face to index")
        return r["FaceRecords"][0]["Face"]["FaceId"]

    def delete_faces(self, face_ids: list[str]) -> None:
        if face_ids:
            self.client.delete_faces(CollectionId=self.collection, FaceIds=face_ids)
