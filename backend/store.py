"""Attempt log (SQLite under data/, gitignored) + optional saved eval frames under eval_data/<label>/.
Failed enrollments are logged WITHOUT images or thumbnails."""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVAL = ROOT / "eval_data"
LABELS = ("genuine", "impostor", "photo", "screen", "video", "mask", "unlabeled")
ATTACK_LABELS = ("photo", "screen", "video", "mask")


class Store:
    def __init__(self, data_dir: Path = DATA) -> None:
        self.dir = data_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.dir / "bench.db", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, ts REAL, label TEXT, variant TEXT, user TEXT, source TEXT,
            decision TEXT, reason TEXT, raw TEXT, timings TEXT, per_frame TEXT, thumb BLOB, saved_dir TEXT)""")
        self.db.commit()

    # ---------- attempts ----------
    def add_attempt(self, *, label: str, variant: str, user: str | None, source: str, decision: str,
                    reason: str, raw: dict[str, Any], timings: dict[str, float], per_frame: list[dict[str, Any]],
                    thumb: bytes | None, frames: list[bytes] | None) -> str:
        aid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        saved = None
        if frames:
            sd = EVAL / label / aid
            sd.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(frames):
                (sd / f"frame_{i}.jpg").write_bytes(b)
            (sd / "meta.json").write_text(json.dumps({"id": aid, "label": label, "user": user,
                                                      "variant": variant, "ts": time.time()}))
            saved = str(sd.relative_to(ROOT))
        with self.lock:
            self.db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (aid, time.time(), label, variant, user, source, decision, reason, json.dumps(raw),
                             json.dumps(timings), json.dumps(per_frame), thumb, saved))
            self.db.commit()
        return aid

    def attempts(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT id, ts, label, variant, user, source, decision, reason, raw, timings, "
                                   "per_frame, saved_dir, thumb IS NOT NULL FROM attempts ORDER BY ts").fetchall()
        keys = ["id", "ts", "label", "variant", "user", "source", "decision", "reason", "raw", "timings",
                "per_frame", "saved_dir", "has_thumb"]
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            for k in ("raw", "timings", "per_frame"):
                d[k] = json.loads(d[k])
            d["has_thumb"] = bool(d["has_thumb"])
            out.append(d)
        return out

    def set_client_ms(self, aid: str, e2e_ms: float) -> bool:
        """Browser-measured time from end of capture to decision rendered (the brief's <=2 s metric)."""
        with self.lock:
            r = self.db.execute("SELECT timings FROM attempts WHERE id=?", (aid,)).fetchone()
            if not r:
                return False
            t = json.loads(r[0])
            t["client_e2e"] = round(float(e2e_ms), 1)
            self.db.execute("UPDATE attempts SET timings=? WHERE id=?", (json.dumps(t), aid))
            self.db.commit()
        return True

    def thumb(self, aid: str) -> bytes | None:
        with self.lock:
            r = self.db.execute("SELECT thumb FROM attempts WHERE id=?", (aid,)).fetchone()
        return r[0] if r else None

    def delete_attempt(self, aid: str) -> bool:
        with self.lock:
            r = self.db.execute("SELECT saved_dir FROM attempts WHERE id=?", (aid,)).fetchone()
            if not r:
                return False
            self.db.execute("DELETE FROM attempts WHERE id=?", (aid,))
            self.db.commit()
        if r[0]:
            shutil.rmtree(ROOT / r[0], ignore_errors=True)
        return True

    def reset(self, include_eval: bool = False) -> None:
        """Demo safety: wipe attempts (eval_data only if asked). People are reset by People.reset()."""
        with self.lock:
            self.db.execute("DELETE FROM attempts")
            self.db.commit()
        if include_eval:
            shutil.rmtree(EVAL, ignore_errors=True)
