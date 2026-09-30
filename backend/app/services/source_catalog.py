"""Process-local catalog for sources and harmonization results."""

from __future__ import annotations

import threading
import uuid
from typing import Any


class SourceCatalog:
    """Retain ingested records for the lifetime of one backend process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}

    def register(self, source: dict[str, Any]) -> dict[str, Any]:
        source_id = str(source.get("source_id") or uuid.uuid4())
        stored = dict(source)
        stored["source_id"] = source_id
        with self._lock:
            self._sources[source_id] = stored
        return self.source_summary(stored)

    @staticmethod
    def source_summary(source: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in source.items() if key != "records"}

    def list_sources(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.source_summary(source) for source in self._sources.values()]

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        with self._lock:
            source = self._sources.get(source_id)
            return dict(source) if source is not None else None

    def list_layers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    **self.source_summary(source),
                    "layer_id": source["source_id"],
                    "feature_count": len(source.get("records", [])),
                }
                for source in self._sources.values()
            ]

    def ulpin_matches(self, ulpin: str) -> list[dict[str, Any]]:
        target = ulpin.strip().casefold()
        matches = []
        with self._lock:
            sources = list(self._sources.values())
        for source in sources:
            for record in source.get("records", []):
                attributes = record.get("raw_attributes") or {}
                value = record.get("ulpin")
                if value is None:
                    value = next(
                        (
                            item
                            for key, item in attributes.items()
                            if str(key).strip().casefold() in {"ulpin", "unique_land_parcel_id"}
                        ),
                        None,
                    )
                if value is not None and str(value).strip().casefold() == target:
                    matches.append({**record, "source_id": source["source_id"]})
        return matches

    def save_job(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job.get("job_id") or uuid.uuid4())
        stored = dict(job)
        stored["job_id"] = job_id
        with self._lock:
            self._jobs[job_id] = stored
        return dict(stored)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None


source_catalog = SourceCatalog()