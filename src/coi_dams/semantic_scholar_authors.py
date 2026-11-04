"""Streaming conversion of the Semantic Scholar "authors" dump into a Parquet
cache, and identity-mapping helpers.

The authors dump has no publications, but a
fraction of its records carry an `externalids.DBLP` field. This is the DBLP name
spellings that author publishes under. It's used in two ways:

  1. `build_dblp_name_map` turns it into authorid -> [DBLP name, ...], which
     semantic_scholar_papers.py uses to fold S2 paper authors into DBLP names
  2. `enrich_homepages_with_semantic_scholar` folds each such author's S2
     name, aliases and affiliations into the matching DBLP homepage record
     before users.py runs its disambiguation
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import cache
from .constants import S2_PARSE_WORKERS
from .dblp import iter_homepage_records, write_homepages
from .parquet_io import PARTIAL_SUFFIX, merge_parquet_shards, write_records

logger = logging.getLogger(__name__)

SEMSCHOLAR_AUTHOR_SCHEMA = pa.schema(
    [
        pa.field("authorid", pa.string()),
        pa.field("name", pa.string()),
        pa.field("aliases", pa.list_(pa.string())),
        pa.field("affiliations", pa.list_(pa.string())),
        pa.field("homepage", pa.string()),
        pa.field("papercount", pa.int32()),
        pa.field("citationcount", pa.int32()),
        pa.field("hindex", pa.int32()),
        pa.field("dblp_names", pa.list_(pa.string())),
        pa.field("orcid", pa.list_(pa.string())),
    ]
)


# Every `part-*.jsonl.gz` file under `source_dir`
def _part_files(source_dir: str) -> list[Path]:
    return sorted(Path(source_dir).glob("part-*.jsonl.gz"))


def check_source_dir(source_dir: str, kind: str) -> None:
    if not _part_files(source_dir):
        raise SystemExit(
            f"no part-*.jsonl.gz files found under {source_dir} - check the "
            f"{kind} directory path"
        )


def iter_jsonl_gz_dir(source_dir: str) -> Iterator[dict]:
    """Yield every JSON record across every `part-*.jsonl.gz` file in a directory.

    Shared by the authors and papers modules, both dumps use the same layout.
    """
    for part in _part_files(source_dir):
        with gzip.open(part, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


# The dump uses `null` interchangeably with `[]`
def _as_list(value: list | None) -> list[str]:
    return value or []


def _parse_author(raw: dict) -> dict:
    external_ids = raw.get("externalids") or {}
    return {
        "authorid": raw.get("authorid") or "",
        "name": raw.get("name") or "",
        "aliases": _as_list(raw.get("aliases")),
        "affiliations": _as_list(raw.get("affiliations")),
        "homepage": raw.get("homepage") or "",
        "papercount": raw.get("papercount") or 0,
        "citationcount": raw.get("citationcount") or 0,
        "hindex": raw.get("hindex") or 0,
        "dblp_names": _as_list(external_ids.get("DBLP")),
        "orcid": _as_list(external_ids.get("ORCID")),
    }


# Parsed author records from a single part file
def _authors_in_part(part_path: Path) -> Iterator[dict]:
    with gzip.open(part_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield _parse_author(json.loads(line))


def parse_authors(source_dir: str) -> Iterator[dict]:
    """Parsed author records, one per line, across every part file."""
    for raw in iter_jsonl_gz_dir(source_dir):
        yield _parse_author(raw)


# Worker body: parse one part file into its own shard Parquet
def _build_authors_shard(part_path_str: str, shard_path: str) -> int:
    return write_records(
        _authors_in_part(Path(part_path_str)), shard_path, SEMSCHOLAR_AUTHOR_SCHEMA
    )


def parallel_build(
    source_dir: str,
    output_parquet: str,
    worker_fn,
    schema: pa.Schema,
    extra_worker_args: tuple = (),
    manifest: cache.Manifest | None = None,
) -> int:
    """Parse every part file in `source_dir` in parallel, one shard each, then merge.

    The dumps are tens of independent gzip files, so a process pool parses them
    in parallel at the cost of a streamed merge pass at the end.

    The pool is capped at S2_PARSE_WORKERS instead of os.cpu_count(), since each
    spawned worker imports pyarrow on its own.
    """
    parts = _part_files(source_dir)
    shard_dir = Path(output_parquet).parent / (Path(output_parquet).name + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [str(shard_dir / f"{i:03d}.parquet") for i in range(len(parts))]

    max_workers = min(len(parts), S2_PARSE_WORKERS)
    logger.info(
        "Parsing %s part file(s) across %s worker process(es) ...",
        len(parts),
        max_workers,
    )
    try:
        # Spawn instead of fork since forking with pyarrow's IO threads running can cause deadlocks
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(worker_fn, str(part), shard_path, *extra_worker_args)
                for part, shard_path in zip(parts, shard_paths, strict=True)
            ]
            shard_counts = [future.result() for future in futures]
        logger.info(
            "Parsed %s record(s) across %s shard(s); merging ...",
            f"{sum(shard_counts):,}",
            len(shard_paths),
        )
        return merge_parquet_shards(shard_paths, output_parquet, schema, manifest)
    finally:
        for shard_path in shard_paths:
            Path(shard_path).unlink(missing_ok=True)
        # Non-empty only if a worker left something behind, which is harmless
        with contextlib.suppress(OSError):
            shard_dir.rmdir()


def stream_semantic_scholar_authors(
    source_dir: str, output_parquet: str, force: bool = False
) -> None:
    """Parse the S2 authors dump directory into `output_parquet`.

    Skipped when the Parquet is already built from these part files, and
    written to a `.partial` path first, like dblp.stream_dblp. Each part file is
    parsed by its own worker process (see parallel_build).

    No cutoff year: nothing in the authors dump is filtered by year, so the
    cache stays valid across a cutoff change.
    """
    check_source_dir(source_dir, "--s2-authors-dir")
    if not force and not cache.is_stale(output_parquet, [source_dir]):
        logger.info("S2 authors Parquet up-to-date, skipping parse")
        return

    logger.info("Parsing S2 authors dump at %s ...", source_dir)
    partial = output_parquet + PARTIAL_SUFFIX
    try:
        count = parallel_build(
            source_dir,
            partial,
            _build_authors_shard,
            SEMSCHOLAR_AUTHOR_SCHEMA,
            manifest=cache.build_manifest([source_dir]),
        )
        os.replace(partial, output_parquet)
    finally:
        Path(partial).unlink(missing_ok=True)
    logger.info("Wrote %s S2 author record(s) -> %s", f"{count:,}", output_parquet)


def load_semantic_scholar_authors(parquet_path: str) -> Iterator[dict]:
    """Stream back the parsed author records from a Parquet built above."""
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches():
        yield from batch.to_pylist()


def build_dblp_name_map(authors: Iterator[dict]) -> dict[str, list[str]]:
    """authorid -> the DBLP name spellings that author is known to publish under.

    Only authors with a non-empty externalids.DBLP carry an entry, the rest get
    a synthetic S2-only identity in semantic_scholar_papers.py.
    """
    return {
        author["authorid"]: author["dblp_names"]
        for author in authors
        if author["dblp_names"]
    }


def enriched_homepages_path(homepages_parquet: str) -> str:
    """Where the S2-enriched copy of a homepage index lives."""
    return str(Path(homepages_parquet).with_suffix(".enriched.parquet"))


def build_enriched_homepages(
    homepages_parquet: str,
    authors_parquet: str,
    force: bool = False,
) -> str:
    """Enrich a homepage index with S2 identity data, once; return its path.

    The enrichment streams the whole S2 authors cache (100M+ records), so the
    result is cached under its own manifest.

    The enriched index is a separate artefact, so dropping --s2-authors-dir from
    a later run doesn't keep using S2-derived aliases.
    """
    output = enriched_homepages_path(homepages_parquet)
    sources = [homepages_parquet, authors_parquet]
    if not force and not cache.is_stale(output, sources):
        logger.info("Enriched homepage index up-to-date, skipping S2 enrichment")
        return output

    logger.info("Folding S2 identity data into the DBLP homepage index ...")
    homepages = {
        record["pid"]: record for record in iter_homepage_records(homepages_parquet)
    }
    enriched = enrich_homepages_with_semantic_scholar(
        homepages, load_semantic_scholar_authors(authors_parquet)
    )

    partial = output + PARTIAL_SUFFIX
    try:
        write_homepages(enriched, partial, manifest=cache.build_manifest(sources))
        os.replace(partial, output)
    finally:
        Path(partial).unlink(missing_ok=True)
    logger.info(
        "Wrote %s enriched homepage record(s) -> %s", f"{len(enriched):,}", output
    )
    return output


def enrich_homepages_with_semantic_scholar(
    homepages: dict[str, dict], authors: Iterator[dict]
) -> dict[str, dict]:
    """Fold S2 identity data into the DBLP homepage records it corroborates.

    For every S2 author whose externalids.DBLP matches an alias on a DBLP
    homepage record, the author's S2 name, aliases and affiliations are merged
    into that record, as in dblp.fold_crossref_aliases.

    When a DBLP name appears on more than one record, every such record is
    enriched, there is no stronger signal to pick one.

    The records in `homepages` are mutated and returned.
    """
    alias_to_records: dict[str, list[dict]] = {}
    for record in homepages.values():
        if record.get("crossref"):
            continue
        for alias in record.get("aliases", []):
            alias_to_records.setdefault(alias, []).append(record)

    matched = 0
    for author in authors:
        targets: set[int] = set()
        records: list[dict] = []
        for dblp_name in author["dblp_names"]:
            for record in alias_to_records.get(dblp_name, []):
                if id(record) not in targets:
                    targets.add(id(record))
                    records.append(record)
        if not records:
            continue
        matched += 1
        extra_aliases = {author["name"], *author["aliases"]} - {""}
        extra_affiliations = set(author["affiliations"])
        for record in records:
            record["aliases"] = sorted(set(record["aliases"]) | extra_aliases)
            record["affiliations"] = sorted(
                set(record.get("affiliations", [])) | extra_affiliations
            )

    logger.info(
        "%s S2 author(s) corroborated against an existing DBLP homepage record",
        f"{matched:,}",
    )
    return homepages
