"""Enrolled people: metadata in SQLite, reference images in S3 (SSE-S3, 3-day lifecycle), face vectors in
a Rekognition collection. Deleting a person removes all three. Reference bytes are cached in memory
so the verify hot path never waits on S3."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import boto3

from .matcher import RekognitionMatcher

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class People:
    def __init__(self, db_path: Path, matcher: RekognitionMatcher | None, bucket: str | None,
                 region: str = "us-east-1") -> None:
        self.matcher, self.bucket = matcher, bucket
        self.s3 = boto3.client("s3", region_name=region) if bucket else None
        self.lock = threading.Lock()
        self.cache: dict[str, bytes] = {}
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS people (
            person_id TEXT PRIMARY KEY, name TEXT, role TEXT, enrolled_at REAL, face_id TEXT,
            s3_keys TEXT, ref_index INTEGER, gated INTEGER, consent_ts REAL, photo BLOB)""")
        self.db.commit()

    # ---------- reads ----------
    def ids(self) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT person_id FROM people ORDER BY enrolled_at")]

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT person_id, name, role, enrolled_at, gated, s3_keys FROM people "
                                   "ORDER BY enrolled_at DESC").fetchall()
        return [{"person_id": r[0], "name": r[1], "role": r[2], "enrolled_at": r[3], "liveness_gated": bool(r[4]),
                 "n_images": len(json.loads(r[5]))} for r in rows]

    def get(self, pid: str) -> dict[str, Any] | None:
        return next((p for p in self.list() if p["person_id"] == pid), None)

    def name(self, pid: str) -> str:
        p = self.get(pid)
        return p["name"] if p else pid

    def photo(self, pid: str) -> bytes | None:
        with self.lock:
            r = self.db.execute("SELECT photo FROM people WHERE person_id=?", (pid,)).fetchone()
        return r[0] if r else None

    def reference(self, pid: str) -> bytes | None:
        if pid in self.cache:
            return self.cache[pid]
        with self.lock:
            r = self.db.execute("SELECT s3_keys, ref_index FROM people WHERE person_id=?", (pid,)).fetchone()
        if not r or not self.s3:
            return None
        key = json.loads(r[0])[r[1]]
        self.cache[pid] = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return self.cache[pid]

    def duplicate_of(self, probe: bytes, pid: str, threshold: float) -> tuple[str, float] | None:
        """Another enrolled ID whose face matches the probe above `threshold`, if any."""
        if not self.matcher:
            return None
        hits, _, _ = self.matcher.search(probe, max_faces=5)
        return next(((h, s) for h, s in hits if h != pid and s >= threshold), None)

    # ---------- writes ----------
    def add(self, *, pid: str, name: str, role: str, jpegs: list[bytes], ref_index: int, gated: bool,
            consent_ts: float | None, photo: bytes) -> dict[str, float]:
        """Replaces any existing record for pid. Returns per-step ms."""
        tm: dict[str, float] = {}
        if self.get(pid):
            t0 = time.perf_counter(); self.delete(pid); tm["replace_old"] = (time.perf_counter() - t0) * 1e3  # noqa: E702
        t0 = time.perf_counter()
        face_id = self.matcher.index(pid, jpegs[ref_index]) if self.matcher else None
        tm["index_faces"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        keys = []
        for i, b in enumerate(jpegs):
            key = f"people/{pid}/{int(time.time())}_{i}.jpg"
            if self.s3:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=b, ContentType="image/jpeg",
                                   ServerSideEncryption="AES256")
            keys.append(key)
        tm["s3_put"] = (time.perf_counter() - t0) * 1e3
        with self.lock:
            self.db.execute("INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (pid, name, role, time.time(), face_id, json.dumps(keys), ref_index, int(gated),
                             consent_ts, photo))
            self.db.commit()
        self.cache[pid] = jpegs[ref_index]
        return {k: round(v, 1) for k, v in tm.items()}

    def delete(self, pid: str) -> bool:
        with self.lock:
            r = self.db.execute("SELECT face_id, s3_keys FROM people WHERE person_id=?", (pid,)).fetchone()
        if not r:
            return False
        if r[0] and self.matcher:
            self.matcher.delete_faces([r[0]])
        if self.s3:
            keys = json.loads(r[1])
            if keys:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in keys]})
        with self.lock:
            self.db.execute("DELETE FROM people WHERE person_id=?", (pid,))
            self.db.commit()
        self.cache.pop(pid, None)
        return True

    def reset(self) -> None:
        for pid in self.ids():
            self.delete(pid)
