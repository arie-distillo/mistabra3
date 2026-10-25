"""SQLite store: transactions, append-only audit with a hash chain."""
from __future__ import annotations
import hashlib, json, sqlite3, threading, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)


class Tx:
    def __init__(self, conn: sqlite3.Connection, actor: str, blind: bool):
        self.conn, self.actor, self.blind = conn, actor, blind

    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)

    def one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def audit(self, action: str, object_type: str, object_id: str | None, before=None, after=None, llm_call_id: str | None = None) -> str:
        prev = self.one("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1")
        prev_hash = prev["hash"] if prev else ""
        aid, ts = new_id(), now()
        b = dumps(before) if before is not None else None
        a = dumps(after) if after is not None else None
        h = hashlib.sha256(f"{prev_hash}|{ts}|{self.actor}|{action}|{object_type}|{object_id}|{b}|{a}".encode()).hexdigest()
        self.execute("INSERT INTO audit(id,ts,actor,action,object_type,object_id,before_json,after_json,llm_call_id,blind,prev_hash,hash)"
                     " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (aid, ts, self.actor, action, object_type, object_id, b, a, llm_call_id, int(self.blind), prev_hash, h))
        return aid


class Store:
    def __init__(self, data_dir: Path, blind: bool = True):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "mistabra.sqlite"
        self.blind = blind
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self._lock:
            self.conn.executescript((Path(__file__).parent / "schema.sql").read_text())

    @contextmanager
    def tx(self, actor: str):
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield Tx(self.conn, actor, self.blind)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def one(self, sql, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def all(self, sql, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def count(self, table: str) -> int:
        return self.one(f"SELECT COUNT(*) AS n FROM {table}")["n"]

    def verify_audit(self) -> bool:
        prev = ""
        for r in self.all("SELECT * FROM audit ORDER BY seq"):
            h = hashlib.sha256(f"{prev}|{r['ts']}|{r['actor']}|{r['action']}|{r['object_type']}|{r['object_id']}|{r['before_json']}|{r['after_json']}".encode()).hexdigest()
            if h != r["hash"] or r["prev_hash"] != prev:
                return False
            prev = r["hash"]
        return True

    def close(self):
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.ProgrammingError:
                pass
