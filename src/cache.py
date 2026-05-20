from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class JsonCache:
    path: Path

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_fresh(self, max_age_seconds: int) -> bool:
        if not self.path.exists():
            return False
        mtime = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        age = (datetime.now(tz=timezone.utc) - mtime).total_seconds()
        return age <= max_age_seconds

