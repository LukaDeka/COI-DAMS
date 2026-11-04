"""Streaming conversion of dblp.xml into a Parquet of publications and a homepage index.

The dump does not fit in memory, so it is parsed with iterparse and flushed in chunks
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree

from . import cache
from .constants import DBLP_SCHEMA, HOMEPAGE_SCHEMA, PUB_TAGS
from .parquet_io import PARTIAL_SUFFIX, write_records

logger = logging.getLogger(__name__)

RECORD_PUBLICATION = "pub"
RECORD_HOMEPAGE = "home"

HOMEPAGE_KEY_PREFIX = "homepages/"


def dump_cache_path(xml_path: str) -> str:
    return str(Path(xml_path).with_suffix(".parquet"))


def homepages_path(parquet_path: str) -> str:
    return str(Path(parquet_path).with_suffix(".homepages.parquet"))


def iter_homepage_records(homepages_parquet: str) -> Iterator[dict]:
    """Stream the homepage records of an index without holding them all."""
    parquet = pq.ParquetFile(homepages_parquet)
    for batch in parquet.iter_batches():
        yield from batch.to_pylist()


def canonical_homepage_records(homepages_parquet: str) -> Iterator[dict]:
    """Stream only the homepage records that carry data, skipping pointers.

    The streaming counterpart of canonical_records
    """
    return (
        record
        for record in iter_homepage_records(homepages_parquet)
        if not record.get("crossref")
    )


def load_homepages(homepages_parquet: str) -> dict[str, dict]:
    """The whole homepage index, keyed by pid.

    Loaded fully since users.disambiguate needs random access by pid.
    """
    return {
        record["pid"]: record for record in iter_homepage_records(homepages_parquet)
    }


def write_homepages(
    homepages: dict[str, dict],
    output_parquet: str,
    manifest: cache.Manifest | None = None,
) -> int:
    """Write a homepage index out, manifest and all."""
    return write_records(
        homepages.values(), output_parquet, HOMEPAGE_SCHEMA, manifest=manifest
    )


def canonical_pid(crossref: str) -> str:
    """The pid a crossref pointer ("homepages/12/3456") refers to."""
    return crossref[len(HOMEPAGE_KEY_PREFIX) :]


def canonical_records(homepages: dict[str, dict]) -> Iterator[dict]:
    """Every homepage record that carries data, skipping merged-profile pointers.

    A pointer's aliases are already copied into its target, so indexes must
    skip them or one person is listed under two records.
    """
    return (record for record in homepages.values() if not record.get("crossref"))


def dtd_path(xml_path: str) -> str:
    """Where dblp.dtd is expected: DBLP's own dump references it by this relative name."""
    return str(Path(xml_path).parent / "dblp.dtd")


def _check_xml_exists(xml_path: str) -> None:
    if xml_path.endswith(".gz"):
        raise SystemExit(
            f"--dblp-xml must point at the extracted dump, not {xml_path}. "
            "Run misc/fetch_data_sources.py, which downloads and unpacks it"
        )
    if not Path(xml_path).exists():
        raise SystemExit(
            f"dblp.xml not found at {xml_path}. Check --dblp-xml, or fetch the "
            "dump with misc/fetch_data_sources.py"
        )


# Fails early, a missing dblp.dtd otherwise shows up as XMLSyntaxError minutes into the parse
def _check_dtd_exists(xml_path: str) -> None:
    dtd = dtd_path(xml_path)
    if not Path(dtd).exists():
        raise SystemExit(
            f"dblp.dtd not found at {dtd}. It must be beside {xml_path} and "
            "is needed to resolve entities in the dump. "
            "misc/fetch_data_sources.py downloads both together; otherwise take it "
            "from https://dblp.org/xml/dblp.dtd"
        )


def cache_params(min_year: int) -> dict[str, int]:
    """The settings a publications cache's contents depend on.

    Only the cutoff year, since it decides which records exist. Chunk size and
    worker count don't change the contents.
    """
    return {cache.PARAM_CUTOFF_YEAR: min_year}


# Free an element and every preceding sibling already parsed
def _drop_parsed_element(elem: etree._Element) -> None:
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]


# Yields direct children of <dblp>, then frees them.
# The DTD is loaded, so only use this on a trusted dump (XXE)
def _iter_dblp_children(xml_path: str) -> Iterator[etree._Element]:
    context = etree.iterparse(
        xml_path, load_dtd=True, resolve_entities=True, events=("end",)
    )
    for _event, elem in context:
        parent = elem.getparent()
        if parent is None or parent.tag != "dblp":
            continue
        try:
            yield elem
        finally:
            _drop_parsed_element(elem)
    del context
    gc.collect()


def _texts(elem: etree._Element, tag: str) -> list[str]:
    return [child.text for child in elem.findall(tag) if child.text]


# Parse a <www> homepage record, or return None for any other <www>
def _parse_homepage(elem: etree._Element) -> dict | None:
    key = elem.get("key", "")
    if not key.startswith(HOMEPAGE_KEY_PREFIX):
        return None

    pid = key[len(HOMEPAGE_KEY_PREFIX) :]
    # A crossref record is one half of a merged profile, only its aliases are kept for folding
    crossref = elem.findtext("crossref")
    if crossref:
        return {
            "pid": pid,
            "crossref": crossref,
            "aliases": _texts(elem, "author"),
            "affiliations": [],
            "urls": [],
        }

    return {
        "pid": pid,
        "crossref": None,
        "aliases": _texts(elem, "author"),
        "affiliations": [
            note.text
            for note in elem.findall("note")
            if note.get("type") == "affiliation" and note.text
        ],
        "urls": _texts(elem, "url"),
    }


def _parse_year(elem: etree._Element) -> int:
    try:
        return int(elem.findtext("year") or 0)
    except ValueError:
        return 0


# Best-available DOI: a doi.org <ee> link, else a bare <doi> made into one
def _parse_doi(elem: etree._Element) -> str:
    for ee in elem.findall("ee"):
        link = (ee.text or "").strip()
        if "doi.org/" in link:
            return link
    bare_doi = elem.findtext("doi")
    return f"https://doi.org/{bare_doi.strip()}" if bare_doi else ""


# Parse a publication element, or return None if it predates `min_year`
def _parse_publication(elem: etree._Element, min_year: int) -> dict | None:
    year = _parse_year(elem)
    if year < min_year:
        return None
    return {
        "key": elem.get("key", ""),
        "title": elem.findtext("title") or "",
        "authors": _texts(elem, "author"),
        "year": year,
        "venue": elem.findtext("journal") or elem.findtext("booktitle") or "",
        "pub_type": elem.tag,
        "doi": _parse_doi(elem),
    }


# Yields ("pub" | "home", record) in one pass, a second pass over the XML would cost as much
def _iter_records(xml_path: str, min_year: int) -> Iterator[tuple[str, dict]]:
    for elem in _iter_dblp_children(xml_path):
        if elem.tag == "www":
            homepage = _parse_homepage(elem)
            if homepage is not None:
                yield RECORD_HOMEPAGE, homepage
        elif elem.tag in PUB_TAGS:
            publication = _parse_publication(elem, min_year)
            if publication is not None:
                yield RECORD_PUBLICATION, publication


def fold_crossref_aliases(raw_homepages: dict[str, dict]) -> dict[str, dict]:
    """Copy each crossref (merged-profile) record's aliases into its target.

    DBLP keeps both pids of a merged author, one pointing at the other, so a
    lookup by either pid finds every name.

    The pointer records are kept in the result. A Users.csv row may cite the
    secondary pid, which would otherwise resolve to nothing and usually end up
    DISAM_FAILED.
    """
    for record in raw_homepages.values():
        if not record["crossref"]:
            continue
        target = raw_homepages.get(canonical_pid(record["crossref"]))
        if target is not None and not target["crossref"]:
            target["aliases"] = sorted(set(target["aliases"]) | set(record["aliases"]))
    return raw_homepages


def stream_dblp(
    xml_path: str, output_parquet: str, min_year: int, force: bool = False
) -> None:
    """Parse dblp.xml into `output_parquet` plus its homepage index.

    Skipped when both artefacts are already built from this dump. Refused when
    either was built under a different cutoff year.

    Both are written to `.partial` files and renamed into place once the parse
    has finished, so a crash doesn't leave an invalid Parquet.
    """
    _check_xml_exists(xml_path)
    homepages_parquet = homepages_path(output_parquet)
    params = cache_params(min_year)
    sources = [xml_path]

    for path, description in (
        (output_parquet, "The DBLP publications cache"),
        (homepages_parquet, "The DBLP homepage index"),
    ):
        cache.check_parameters(path, params, force=force, description=description)

    if (
        not force
        and not cache.is_stale(output_parquet, sources)
        and not cache.is_stale(homepages_parquet, sources)
    ):
        logger.info("Parquet and homepages up-to-date, skipping parse")
        return

    _check_dtd_exists(xml_path)
    logger.info(
        "Parsing %s (year >= %s), takes 5-10 min for the entire DBLP ...",
        xml_path,
        min_year,
    )
    manifest = cache.build_manifest(sources, params)
    parquet_partial = output_parquet + PARTIAL_SUFFIX
    homepages_partial = homepages_parquet + PARTIAL_SUFFIX
    raw_homepages: dict[str, dict] = {}

    def publications() -> Iterator[dict]:
        for kind, record in _iter_records(xml_path, min_year):
            if kind == RECORD_HOMEPAGE:
                raw_homepages[record["pid"]] = record
            else:
                yield record

    try:
        # Every chunk leaves a cleared lxml element tree behind, so collect on each flush
        publication_count = write_records(
            publications(),
            parquet_partial,
            DBLP_SCHEMA,
            gc_interval=1,
            manifest=manifest,
        )
        logger.info("Parsed %s pub records", f"{publication_count:,}")

        homepages = fold_crossref_aliases(raw_homepages)
        pointer_count = sum(1 for record in homepages.values() if record["crossref"])
        write_homepages(homepages, homepages_partial, manifest=manifest)

        os.replace(parquet_partial, output_parquet)
        os.replace(homepages_partial, homepages_parquet)
    finally:
        for partial in (parquet_partial, homepages_partial):
            Path(partial).unlink(missing_ok=True)
        gc.collect()

    logger.info("Wrote %s pub records -> %s", f"{publication_count:,}", output_parquet)
    logger.info(
        "Wrote %s homepage records (%s of them merged-profile pointers) -> %s",
        f"{len(homepages):,}",
        f"{pointer_count:,}",
        homepages_parquet,
    )
