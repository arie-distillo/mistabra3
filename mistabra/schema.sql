CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL, ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  object_type TEXT NOT NULL, object_id TEXT, before_json TEXT, after_json TEXT, llm_call_id TEXT, blind INTEGER NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_audit_obj ON audit(object_type, object_id);

CREATE TABLE IF NOT EXISTS llm_call (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, task TEXT NOT NULL, prompt_version TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  input_hash TEXT NOT NULL, input_json TEXT NOT NULL, output_text TEXT, ok INTEGER NOT NULL, cached INTEGER NOT NULL,
  latency_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER, error TEXT, subject_type TEXT, subject_id TEXT);
CREATE INDEX IF NOT EXISTS ix_llm_subject ON llm_call(subject_type, subject_id);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, output_text TEXT NOT NULL, ts TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS context_item (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT, text TEXT NOT NULL, tags_json TEXT NOT NULL, origin TEXT NOT NULL,
  ref_type TEXT, ref_id TEXT, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS risk (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, polarity TEXT NOT NULL, status TEXT NOT NULL,
  entities_json TEXT NOT NULL, origin TEXT NOT NULL, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS radar (
  id TEXT PRIMARY KEY, risk_id TEXT NOT NULL, query TEXT NOT NULL, enabled INTEGER NOT NULL, origin TEXT NOT NULL,
  created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS source (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, type TEXT NOT NULL, credibility REAL NOT NULL, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS episode (
  id TEXT PRIMARY KEY, source_id TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
  occurred_kind TEXT NOT NULL, occurred_start TEXT, occurred_end TEXT, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(source_id, content_hash));

CREATE TABLE IF NOT EXISTS signal (
  id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, risk_id TEXT NOT NULL, origin TEXT NOT NULL, via TEXT NOT NULL, radar_id TEXT,
  content_json TEXT NOT NULL, note TEXT, strength_system REAL, strength_analyst REAL, metrics_json TEXT, missing_json TEXT NOT NULL,
  status TEXT NOT NULL, versions_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_signal_risk ON signal(risk_id);
CREATE INDEX IF NOT EXISTS ix_signal_episode ON signal(episode_id);

CREATE TABLE IF NOT EXISTS hypothesis (
  id TEXT PRIMARY KEY, risk_id TEXT NOT NULL, claim TEXT NOT NULL, entities_json TEXT NOT NULL, status TEXT NOT NULL,
  origin TEXT NOT NULL, is_lambda INTEGER NOT NULL DEFAULT 0, predicts_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS trajectory (
  id TEXT PRIMARY KEY, signal_id TEXT NOT NULL, episode_id TEXT NOT NULL, risk_id TEXT NOT NULL, radar_id TEXT,
  hypothesis_id TEXT NOT NULL, outcome TEXT NOT NULL, ordinary_outcome TEXT, reach REAL NOT NULL, fit REAL NOT NULL,
  rationale TEXT NOT NULL, status TEXT NOT NULL, origin TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_traj_signal ON trajectory(signal_id);
