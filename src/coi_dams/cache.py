from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

METADATA_KEY = b"coi_dams"

FORMAT_VERSION = 1

PARAM_CUTOFF_YEAR = "COI_OLDEST_CUTOFF_YEAR"


class SourceFingerprint(NamedTuple):

    path: str
    size: int
    mtime: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "mtime": self.mtime}


class Manifest(NamedTuple):
    """What a cached artefact was built from, and with what settings"""

    format_version: int
    params: dict[str, Any]
    sources: list[dict[str, Any]]
    built_at: str

    def to_json(self) -> bytes:
        return json.dumps(self._asdict(), sort_keys=True).encode("utf-8")


class ParamMismatch(NamedTuple):
    key: str
    cached: Any
    requested: Any


def fingerprint(paths: list[str]) -> list[dict[str, Any]]:
    """Fingerprint every input path, expanding a directory into its files, sorted by path"""
    prints: list[SourceFingerprint] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files = [path]
        else:
            continue
        for file in files:
            stat = file.stat()
            prints.append(
                SourceFingerprint(str(file), stat.st_size, int(stat.st_mtime))
            )
    return [p.as_dict() for p in sorted(prints)]


def build_manifest(
    sources: list[str], params: dict[str, Any] | None = None
) -> Manifest:
    return Manifest(
        format_version=FORMAT_VERSION,
        params=dict(params or {}),
        sources=fingerprint(sources),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def schema_with_manifest(schema: pa.Schema, manifest: Manifest | None) -> pa.Schema:
    if manifest is None:
        return schema
    return schema.with_metadata(
        {**(schema.metadata or {}), METADATA_KEY: manifest.to_json()}
    )


def read_manifest(parquet_path: str) -> Manifest | None:
    """The manifest a cached Parquet has.

    None when the file is absent, unreadable, different format version, etc.
    """
    if not Path(parquet_path).exists():
        return None
    try:
        metadata = pq.ParquetFile(parquet_path).schema_arrow.metadata or {}
        raw = metadata.get(METADATA_KEY)
        if raw is None:
            return None
        loaded = json.loads(raw)
        manifest = Manifest(
            format_version=loaded["format_version"],
            params=loaded["params"],
            sources=loaded["sources"],
            built_at=loaded["built_at"],
        )
    except (OSError, ValueError, KeyError, pa.ArrowInvalid):
        logger.warning("Could not read the cache manifest of %s", parquet_path)
        return None
    if manifest.format_version != FORMAT_VERSION:
        return None
    return manifest


def mismatched_params(
    manifest: Manifest, params: dict[str, Any]
) -> list[ParamMismatch]:
    """Only compare keys from `params`, extra keys in the cache are ignored."""
    return [
        ParamMismatch(key, manifest.params.get(key), requested)
        for key, requested in params.items()
        if manifest.params.get(key) != requested
    ]


def _describe(mismatches: list[ParamMismatch]) -> str:
    return "\n".join(
        f"  {m.key}: cache was built with {m.cached!r}, this run wants {m.requested!r}"
        for m in mismatches
    )


def check_parameters(
    parquet_path: str,
    params: dict[str, Any],
    *,
    force: bool,
    description: str,
) -> None:
    """Refuse to reuse a cache built under different settings.

    Raises SystemExit instead of rebuilding silently to not go unnoticed.
    If `force` is set, skip
    """
    if force or not params or not Path(parquet_path).exists():
        return

    manifest = read_manifest(parquet_path)
    if manifest is None:
        raise SystemExit(
            f"{parquet_path} carries no readable cache manifest, so the "
            f"settings it was built under ({', '.join(params)}) cannot be "
            "verified.\n"
            "It predates manifests or was written by a different version of "
            "this tool. Rebuild it once by adding --force to your usual "
            "command; later runs reuse the rebuilt cache and need no flag."
        )

    mismatches = mismatched_params(manifest, params)
    if not mismatches:
        return

    raise SystemExit(
        f"{description} ({parquet_path}) was built under different settings:\n"
        f"{_describe(mismatches)}\n\n"
        "Reusing it would analyse data selected under the cached setting "
        "while the run reports the current one.\n"
        "Either restore the previous value, or rebuild the caches once by "
        "adding --force to your usual command. After that run, drop --force "
        "again - later runs reuse the rebuilt cache."
    )


def is_stale(parquet_path: str, sources: list[str]) -> bool:
    """Whether `parquet_path` must be rebuilt.

    True when the artefact is missing, has no usable manifest, or an input's
    size/mtime changed
    """
    manifest = read_manifest(parquet_path)
    if manifest is None:
        return True
    return manifest.sources != fingerprint(sources)
