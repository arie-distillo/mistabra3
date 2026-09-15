"""SQLite system of record with transactional, append-only audit.

Every state change is made inside `Store.tx(actor)`; the audit event is written in
the same transaction, so no change can exist without its record.
"""
from __future__ import annotations
import atexit, hashlib, json, sqlite3, threading, uuid, weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


class Tx:
    def __init__(self, conn: sqlite3.Connection, actor: str):
        self.conn, self.actor = conn, actor
        self.audit_ids: list[str] = []

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def one(self, sql: str, params=()):
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params=()):
        return self.conn.execute(sql, params).fetchall()

    def audit(self, action: str, object_type: str, object_id: str | None,
              before=None, after=None, llm_call_id: str | None = None) -> str:
        prev = self.one("SELECT hash FROM audit_event ORDER BY rowid DESC LIMIT 1")
        prev_hash = prev["hash"] if prev else ""
        aid, ts = new_id(), now()
        b = json.dumps(before, sort_keys=True, default=str) if before is not None else None
        a = json.dumps(after, sort_keys=True, default=str) if after is not None else None
        h = hashlib.sha256(f"{prev_hash}|{ts}|{self.actor}|{action}|{object_type}|{object_id}|{b}|{a}".encode()).hexdigest()
        self.execute(
            "INSERT INTO audit_event(id,ts,actor,action,object_type,object_id,before_json,after_json,llm_call_id,prev_hash,hash)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)", (aid, ts, self.actor, action, object_type, object_id, b, a, llm_call_id, prev_hash, h))
        self.audit_ids.append(aid)
        return aid


class Store:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "mistabra.sqlite"
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.migrate()
        # Close before interpreter teardown: an open sqlite connection alongside an
        # embedded vector store (LanceDB) can segfault during shutdown otherwise.
        atexit.register(Store._close_ref, weakref.ref(self))

    @staticmethod
    def _close_ref(ref):
        s = ref()
        if s is not None:
            s.close()

    def migrate(self) -> None:
        sql = (Path(__file__).parent / "schema.sql").read_text()
        with self._lock:
            self.conn.executescript(sql)
            row = self.conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self.conn.execute("INSERT INTO schema_version(version) VALUES(?)", (SCHEMA_VERSION,))

    @contextmanager
    def tx(self, actor: str):
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            t = Tx(self.conn, actor)
            try:
                yield t
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def one(self, sql: str, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def verify_audit_chain(self) -> bool:
        prev = ""
        for r in self.all("SELECT * FROM audit_event ORDER BY rowid"):
            h = hashlib.sha256(f"{prev}|{r['ts']}|{r['actor']}|{r['action']}|{r['object_type']}|{r['object_id']}|{r['before_json']}|{r['after_json']}".encode()).hexdigest()
            if h != r["hash"] or r["prev_hash"] != prev:
                return False
            prev = r["hash"]
        return True

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.ProgrammingError:
                pass
