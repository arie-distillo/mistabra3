CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS audit_event (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  object_type TEXT NOT NULL, object_id TEXT, before_json TEXT, after_json TEXT,
  llm_call_id TEXT, prev_hash TEXT, hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_obj ON audit_event(object_type, object_id);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_event(ts);

CREATE TABLE IF NOT EXISTS llm_call (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, task TEXT NOT NULL, tier TEXT NOT NULL,
  prompt_id TEXT NOT NULL, prompt_version TEXT NOT NULL, model TEXT NOT NULL, provider TEXT NOT NULL,
  input_hash TEXT NOT NULL, input_json TEXT NOT NULL, output_text TEXT, parsed_ok INTEGER,
  cached INTEGER NOT NULL DEFAULT 0, latency_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS llm_cache (
  cache_key TEXT PRIMARY KEY, output_text TEXT NOT NULL, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS ix_job_status ON job(status, created_at);

CREATE TABLE IF NOT EXISTS ontology_version (
  id TEXT PRIMARY KEY, ontology_id TEXT NOT NULL, version INTEGER NOT NULL, path TEXT NOT NULL,
  published_at TEXT NOT NULL, published_by TEXT NOT NULL, element_count INTEGER NOT NULL,
  UNIQUE(ontology_id, version)
);
CREATE TABLE IF NOT EXISTS ontology_draft (
  id TEXT PRIMARY KEY, ontology_id TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT NOT NULL,
  status TEXT NOT NULL, base_version INTEGER, published_version_id TEXT
);
CREATE TABLE IF NOT EXISTS ontology_draft_element (
  id TEXT PRIMARY KEY, draft_id TEXT NOT NULL, kind TEXT NOT NULL, element_id TEXT NOT NULL,
  body_json TEXT NOT NULL, status TEXT NOT NULL, origin TEXT NOT NULL, merged_into TEXT,
  UNIQUE(draft_id, kind, element_id)
);

CREATE TABLE IF NOT EXISTS source (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, type TEXT NOT NULL, credibility REAL NOT NULL,
  config_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episode (
  id TEXT PRIMARY KEY, source_id TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
  occurrence_kind TEXT NOT NULL, occurrence_start TEXT, occurrence_end TEXT,
  ingested_at TEXT NOT NULL, metadata_json TEXT NOT NULL, extraction_status TEXT NOT NULL DEFAULT 'pending',
  UNIQUE(source_id, content_hash)
);
CREATE TABLE IF NOT EXISTS observation (
  id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL,
  observable_ids_json TEXT NOT NULL, modifiers_json TEXT NOT NULL, created_at TEXT NOT NULL,
  pipeline_version TEXT NOT NULL, llm_call_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_obs_episode ON observation(episode_id);
CREATE TABLE IF NOT EXISTS entity_mention (
  id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, observation_id TEXT, surface TEXT NOT NULL,
  entity_type TEXT NOT NULL, attributes_json TEXT NOT NULL, entity_id TEXT,
  resolution TEXT NOT NULL DEFAULT 'pending', resolution_score REAL, resolution_rationale TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mention_entity ON entity_mention(entity_id);
CREATE INDEX IF NOT EXISTS ix_mention_episode ON entity_mention(episode_id);
CREATE TABLE IF NOT EXISTS entity (
  id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, canonical_name TEXT NOT NULL, aliases_json TEXT NOT NULL,
  attributes_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', merged_into TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_entity_type_name ON entity(entity_type, canonical_name);
CREATE TABLE IF NOT EXISTS entity_link (
  entity_id TEXT NOT NULL, object_type TEXT NOT NULL, object_id TEXT NOT NULL, role TEXT,
  PRIMARY KEY(entity_id, object_type, object_id)
);

-- Phase-1 schema already carries the M3/M4 objects so migrations stay additive.
CREATE TABLE IF NOT EXISTS hypothesis (
  id TEXT PRIMARY KEY, objective_id TEXT NOT NULL, ontology_version_id TEXT NOT NULL, text TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT NOT NULL, standing REAL
);
CREATE TABLE IF NOT EXISTS assessment (
  id TEXT PRIMARY KEY, hypothesis_id TEXT NOT NULL, observation_id TEXT NOT NULL,
  ontology_version_id TEXT NOT NULL, pipeline_version TEXT NOT NULL, kind TEXT NOT NULL,
  relevance REAL, matched_json TEXT, target_link TEXT, baseline_json TEXT, features_json TEXT,
  severity_proposed REAL, severity_blind REAL, severity_confirmed REAL, rationale TEXT,
  missing_info_json TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_record (
  object_type TEXT NOT NULL, object_id TEXT NOT NULL, embedder TEXT NOT NULL, indexed_at TEXT NOT NULL,
  PRIMARY KEY(object_type, object_id)
);
