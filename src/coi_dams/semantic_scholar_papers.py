"""Streaming conversion of the Semantic Scholar "papers" dump into a Parquet
cache shaped exactly like DBLP_SCHEMA (see publications.py for the merge step
that combines this with DBLP's own cache).

Each paper's `authors` are resolved into DBLP's name space where possible (see
`_resolve_author_name`), so papers link into the existing graph. Authors with
no known DBLP spelling get a synthetic identity instead of being dropped or
merged into an unrelated same-named DBLP author.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path

from . import cache
from .constants import DBLP_SCHEMA
from .dblp import cache_params
from .parquet_io import PARTIAL_SUFFIX, write_records
from .semantic_scholar_authors import (
    check_source_dir,
    iter_jsonl_gz_dir,
    parallel_build,
)

logger = logging.getLogger(__name__)

# S2 publicationtypes -> DBLP pub_type tags, anything else becomes "article"
_S2_PUB_TYPE_MAP = {
    "Conference": "inproceedings",
    "JournalArticle": "article",
    "Book": "book",
    "BookSection": "incollection",
}
_S2_PUB_TYPE_FALLBACK = "article"


def _pub_type(publicationtypes: list[str] | None) -> str:
    for raw_type in publicationtypes or []:
        mapped = _S2_PUB_TYPE_MAP.get(raw_type)
        if mapped:
            return mapped
    return _S2_PUB_TYPE_FALLBACK


def _venue(raw: dict) -> str:
    venue = (raw.get("venue") or "").strip()
    if venue:
        return venue
    journal = raw.get("journal") or {}
    return (journal.get("name") or "").strip()


# A bare DOI becomes a doi.org link, like in dblp._parse_doi
def _doi(external_ids: dict) -> str:
    doi = (external_ids.get("DOI") or "").strip()
    return f"https://doi.org/{doi}" if doi else ""


def synthetic_author_id(author_id: str, name: str) -> str:
    """A stable identity for an S2-only author that can't collide with a DBLP name.

    DBLP names never start with "S2:", and the S2 authorId keeps the node the
    same across papers and runs.
    """
    return f"S2:{author_id}:{name}"


# The author's first DBLP name, else a synthetic S2 identity, None without id and name
def _resolve_author_name(
    author: dict, dblp_name_map: dict[str, list[str]]
) -> str | None:
    author_id = author.get("authorId") or ""
    name = author.get("name") or ""
    dblp_names = dblp_name_map.get(author_id) if author_id else None
    if dblp_names:
        return dblp_names[0]
    if author_id or name:
        return synthetic_author_id(author_id, name)
    return None


def _parse_year(raw: dict) -> int:
    try:
        return int(raw.get("year") or 0)
    except (TypeError, ValueError):
        return 0


def parse_paper(
    raw: dict, dblp_name_map: dict[str, list[str]], min_year: int
) -> dict | None:
    """Parse one S2 paper record into a DBLP_SCHEMA-shaped row, or None.

    Returns None when the paper predates `min_year` or has fewer than two
    resolvable authors, which coi.py would discard anyway.
    """
    year = _parse_year(raw)
    if year < min_year:
        return None

    authors = [
        name
        for name in (
            _resolve_author_name(author, dblp_name_map)
            for author in raw.get("authors") or []
        )
        if name
    ]
    if len(authors) < 2:
        return None

    external_ids = raw.get("externalids") or {}
    return {
        "key": f"s2:{raw.get('corpusid', '')}",
        "title": raw.get("title") or "",
        "authors": authors,
        "year": year,
        "venue": _venue(raw),
        "pub_type": _pub_type(raw.get("publicationtypes")),
        "doi": _doi(external_ids),
    }


def parse_papers(
    source_dir: str, dblp_name_map: dict[str, list[str]], min_year: int
) -> Iterator[dict]:
    for raw in iter_jsonl_gz_dir(source_dir):
        parsed = parse_paper(raw, dblp_name_map, min_year)
        if parsed is not None:
            yield parsed


# Parsed, filtered paper rows from a single part file
def _papers_in_part(
    part_path: Path, dblp_name_map: dict[str, list[str]], min_year: int
) -> Iterator[dict]:
    with gzip.open(part_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parsed = parse_paper(json.loads(line), dblp_name_map, min_year)
            if parsed is not None:
                yield parsed


# Worker that parses one part file into a shard Parquet, must stay picklable
def _build_papers_shard(
    part_path_str: str,
    shard_path: str,
    dblp_name_map: dict[str, list[str]],
    min_year: int,
) -> int:
    return write_records(
        _papers_in_part(Path(part_path_str), dblp_name_map, min_year),
        shard_path,
        DBLP_SCHEMA,
    )


def stream_semantic_scholar_papers(
    source_dir: str,
    output_parquet: str,
    build_name_map: Callable[[], dict[str, list[str]]],
    min_year: int,
    force: bool = False,
    name_map_source: str | None = None,
) -> None:
    """Parse the S2 papers dump directory into `output_parquet`.

    Same cache, .partial and force semantics as stream_semantic_scholar_authors
    and dblp.stream_dblp. Each part file is parsed by its own worker process.

    `build_name_map` is only called when the parse actually runs.

    `name_map_source` is the S2 authors cache the name map comes from. It counts
    as an input, since a rebuilt authors cache changes the resolved names.
    """
    check_source_dir(source_dir, "--s2-papers-dir")
    params = cache_params(min_year)
    sources = [source_dir, *([name_map_source] if name_map_source else [])]

    cache.check_parameters(
        output_parquet, params, force=force, description="The S2 papers cache"
    )
    if not force and not cache.is_stale(output_parquet, sources):
        logger.info("S2 papers Parquet up-to-date, skipping parse")
        return

    dblp_name_map = build_name_map()
    logger.info("Parsing S2 papers dump at %s (year >= %s) ...", source_dir, min_year)
    partial = output_parquet + PARTIAL_SUFFIX
    try:
        count = parallel_build(
            source_dir,
            partial,
            _build_papers_shard,
            DBLP_SCHEMA,
            extra_worker_args=(dblp_name_map, min_year),
            manifest=cache.build_manifest(sources, params),
        )
        os.replace(partial, output_parquet)
    finally:
        Path(partial).unlink(missing_ok=True)
    logger.info("Wrote %s S2 paper record(s) -> %s", f"{count:,}", output_parquet)
