"""In-process job runner backed by the SQLite `job` table (resumable, no broker)."""
from __future__ import annotations
import json, traceback
from typing import Callable
from ..store import Store, new_id, now

Handler = Callable[[dict], None]


class Jobs:
    def __init__(self, store: Store, log: Callable[[str], None] | None = None):
        self.store, self._handlers, self.log = store, {}, (log or (lambda m: None))

    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    def enqueue(self, kind: str, payload: dict, actor: str = "system") -> str:
        jid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO job(id,kind,payload_json,status,created_at) VALUES(?,?,?,?,?)",
                       (jid, kind, json.dumps(payload), "queued", now()))
        return jid

    def pending(self) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM job WHERE status='queued' ORDER BY created_at")]

    def run_pending(self, max_jobs: int | None = None, max_attempts: int = 2) -> dict:
        """Run queued jobs until none remain (jobs may enqueue follow-ups)."""
        done = failed = 0
        while True:
            row = self.store.one("SELECT * FROM job WHERE status='queued' ORDER BY created_at LIMIT 1")
            if row is None or (max_jobs is not None and done + failed >= max_jobs):
                break
            job = dict(row)
            with self.store.tx("system") as tx:
                tx.execute("UPDATE job SET status='running', started_at=?, attempts=attempts+1 WHERE id=?", (now(), job["id"]))
            handler = self._handlers.get(job["kind"])
            try:
                if handler is None:
                    raise RuntimeError(f"no handler for job kind {job['kind']}")
                self.log(f"job {job['kind']} {job['id'][:8]} start")
                handler(json.loads(job["payload_json"]))
                with self.store.tx("system") as tx:
                    tx.execute("UPDATE job SET status='done', finished_at=? WHERE id=?", (now(), job["id"]))
                done += 1
            except Exception as e:  # noqa: BLE001 — failures are recorded, not raised
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
                status = "queued" if job["attempts"] + 1 < max_attempts and _transient(e) else "failed"
                with self.store.tx("system") as tx:
                    tx.execute("UPDATE job SET status=?, finished_at=?, error=? WHERE id=?", (status, now(), err, job["id"]))
                failed += 1
                self.log(f"job {job['kind']} {job['id'][:8]} FAILED: {type(e).__name__}: {e}")
        return {"done": done, "failed": failed}


def _transient(e: Exception) -> bool:
    from ..llm.gateway import TransportError
    return isinstance(e, TransportError)
