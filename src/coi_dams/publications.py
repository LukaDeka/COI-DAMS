"""Generic combine step for publication Parquet caches.

Every source module parses its own dump into a Parquet shaped like DBLP_SCHEMA.
This module only knows that shape, so a new source is just another path in the
list pipeline.py passes here.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import cache
from .constants import DBLP_SCHEMA, XML_PARSE_CHUNK_SIZE
from .parquet_io import PARTIAL_SUFFIX

logger = logging.getLogger(__name__)


def is_combined_cache_stale(combined_path: str, source_paths: list[str]) -> bool:
    """Whether the combined cache must be rebuilt from these sources."""
    return cache.is_stale(combined_path, source_paths)


# Keeps the first row per DOI, rows without a DOI are always kept
def _rows_deduped_by_doi(rows: Iterator[dict], seen_dois: set[str]) -> list[dict]:
    kept = []
    for row in rows:
        doi = row["doi"] or ""
        if doi:
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
        kept.append(row)
    return kept


def merge_and_dedup_parquets(
    source_paths: list[str],
    combined_output_path: str,
    params: dict | None = None,
) -> None:
    """Concatenate every source's Parquet into one, dropping duplicate DOIs.

    Earlier sources in `source_paths` win a DOI collision, so list the most
    authoritative one (DBLP) first. Streamed in bounded batches, a single source
    can be tens of millions of rows.

    Written to a .partial file first, like the source parsers. `params` are
    recorded in the manifest, since the combined cache inherits its sources'
    cutoff year.
    """
    Path(combined_output_path).parent.mkdir(parents=True, exist_ok=True)
    partial = combined_output_path + PARTIAL_SUFFIX
    manifest = cache.build_manifest(source_paths, params)
    seen_dois: set[str] = set()
    total = 0
    dropped = 0
    try:
        writer = pq.ParquetWriter(
            partial,
            schema=cache.schema_with_manifest(DBLP_SCHEMA, manifest),
            compression="snappy",
        )
        try:
            for path in source_paths:
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(batch_size=XML_PARSE_CHUNK_SIZE):
                    rows = batch.to_pylist()
                    kept = _rows_deduped_by_doi(rows, seen_dois)
                    dropped += len(rows) - len(kept)
                    if kept:
                        writer.write_table(
                            pa.Table.from_pylist(kept, schema=DBLP_SCHEMA)
                        )
                        total += len(kept)
                    del rows, kept
                gc.collect()
        finally:
            writer.close()
        os.replace(partial, combined_output_path)
    finally:
        Path(partial).unlink(missing_ok=True)
    logger.info(
        "Merged %s source(s) -> %s rows kept, %s dropped as duplicate DOI -> %s",
        len(source_paths),
        f"{total:,}",
        f"{dropped:,}",
        combined_output_path,
    )
