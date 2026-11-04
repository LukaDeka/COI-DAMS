"""Chunked Parquet writing and merging, shared by every source parser.

Each source module streams a multi-GB dump into a Parquet cache. They share two
primitives: writing an iterator of row dicts in bounded chunks, and
concatenating same-schema Parquets without loading either side.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .cache import Manifest, schema_with_manifest
from .constants import XML_PARSE_CHUNK_SIZE

# Suffix of a half-written artefact, renamed into place once its writer finished
PARTIAL_SUFFIX = ".partial"

# Chunk-writes between gc.collect() calls, collecting every chunk is measurable overhead at S2 scale
DEFAULT_GC_INTERVAL = 25


def write_records(
    records: Iterable[dict],
    output_parquet: str,
    schema: pa.Schema,
    gc_interval: int = DEFAULT_GC_INTERVAL,
    manifest: Manifest | None = None,
) -> int:
    """Write records to Parquet in bounded chunks; return the row count.

    The writer is opened on the first record, so a producer that yields nothing
    leaves no Parquet behind and the artefact is rebuilt next run.
    """
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    write_schema = schema_with_manifest(schema, manifest)
    writer: pq.ParquetWriter | None = None
    chunk: list[dict] = []
    total = 0
    flush_count = 0

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal total, flush_count
        table = pa.Table.from_pylist(chunk, schema=schema)
        writer.write_table(table)
        total += len(chunk)
        chunk.clear()
        # Chunk tables run to hundreds of MB, dropping them keeps peak RSS flat
        del table
        flush_count += 1
        if flush_count % gc_interval == 0:
            gc.collect()

    try:
        for record in records:
            if writer is None:
                writer = pq.ParquetWriter(
                    output_parquet, schema=write_schema, compression="snappy"
                )
            chunk.append(record)
            if len(chunk) >= XML_PARSE_CHUNK_SIZE:
                flush(writer)
        if chunk and writer is not None:
            flush(writer)
    finally:
        if writer is not None:
            writer.close()
    return total


def merge_parquet_shards(
    shard_paths: list[str],
    output_parquet: str,
    schema: pa.Schema,
    manifest: Manifest | None = None,
) -> int:
    """Concatenate several same-schema Parquet files into one, streamed.

    Reads each shard in bounded batches so peak memory stays flat. Only the
    merged output carries the `manifest`.
    """
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    writer = pq.ParquetWriter(
        output_parquet,
        schema=schema_with_manifest(schema, manifest),
        compression="snappy",
    )
    try:
        for shard_path in shard_paths:
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(batch_size=XML_PARSE_CHUNK_SIZE):
                table = pa.Table.from_batches([batch], schema=schema)
                writer.write_table(table)
                total += table.num_rows
                del table
            gc.collect()
    finally:
        writer.close()
    return total
