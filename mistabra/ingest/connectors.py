"""Minimal connectors: each yields dicts {text, occurrence_time?, metadata?}."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: Path | str) -> Iterator[dict]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def read_text_files(folder: Path | str, glob: str = "*.txt") -> Iterator[dict]:
    for p in sorted(Path(folder).glob(glob)):
        yield {"text": p.read_text(encoding="utf-8"), "metadata": {"file": p.name}}
