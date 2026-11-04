from __future__ import annotations

import gzip
import json
import logging

import pytest

from coi_dams.cache import is_stale
from coi_dams.dblp import load_homepages, write_homepages
from coi_dams.semantic_scholar_authors import (
    build_dblp_name_map,
    build_enriched_homepages,
    enrich_homepages_with_semantic_scholar,
    parse_authors,
    stream_semantic_scholar_authors,
)


def _write_part(tmp_path, name: str, records: list[dict]):
    source_dir = tmp_path / "authors"
    source_dir.mkdir(exist_ok=True)
    with gzip.open(source_dir / name, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return source_dir


RAW_KUN_ZHANG = {
    "authorid": "143972713",
    "externalids": {"DBLP": ["Kun Zhang 0006", "Kun Zhang 0016"], "ORCID": None},
    "url": "https://www.semanticscholar.org/author/143972713",
    "name": "Kun Zhang",
    "aliases": None,
    "affiliations": None,
    "homepage": None,
    "papercount": 24,
    "citationcount": 122,
    "hindex": 7,
}

RAW_UNMAPPED = {
    "authorid": "999",
    "externalids": {"DBLP": None, "ORCID": ["0000-0001-9529-3317"]},
    "url": "https://www.semanticscholar.org/author/999",
    "name": "Somebody Else",
    "aliases": ["S. Else"],
    "affiliations": ["Some University"],
    "homepage": "https://example.org",
    "papercount": 1,
    "citationcount": 0,
    "hindex": 0,
}


def test_parse_authors_nulls(tmp_path):
    source_dir = _write_part(
        tmp_path, "part-01.jsonl.gz", [RAW_KUN_ZHANG, RAW_UNMAPPED]
    )
    parsed = list(parse_authors(str(source_dir)))

    assert parsed[0] == {
        "authorid": "143972713",
        "name": "Kun Zhang",
        "aliases": [],
        "affiliations": [],
        "homepage": "",
        "papercount": 24,
        "citationcount": 122,
        "hindex": 7,
        "dblp_names": ["Kun Zhang 0006", "Kun Zhang 0016"],
        "orcid": [],
    }
    assert parsed[1]["dblp_names"] == []
    assert parsed[1]["orcid"] == ["0000-0001-9529-3317"]
    assert parsed[1]["aliases"] == ["S. Else"]


def test_parse_authors_all_parts(tmp_path):
    source_dir = _write_part(tmp_path, "part-01.jsonl.gz", [RAW_KUN_ZHANG])
    with gzip.open(source_dir / "part-02.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(RAW_UNMAPPED) + "\n")

    parsed = list(parse_authors(str(source_dir)))
    assert {p["authorid"] for p in parsed} == {"143972713", "999"}


def test_name_map_mapped_only(tmp_path):
    source_dir = _write_part(
        tmp_path, "part-01.jsonl.gz", [RAW_KUN_ZHANG, RAW_UNMAPPED]
    )
    name_map = build_dblp_name_map(parse_authors(str(source_dir)))
    assert name_map == {"143972713": ["Kun Zhang 0006", "Kun Zhang 0016"]}


def test_s2_authors_skips_when_fresh(tmp_path, caplog):
    source_dir = _write_part(tmp_path, "part-01.jsonl.gz", [RAW_KUN_ZHANG])
    parquet_path = str(tmp_path / "s2_authors.parquet")
    stream_semantic_scholar_authors(str(source_dir), parquet_path)

    assert not is_stale(parquet_path, [str(source_dir)])
    with caplog.at_level(logging.INFO):
        stream_semantic_scholar_authors(str(source_dir), parquet_path)
    assert "skipping parse" in caplog.text


def test_s2_authors_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="--s2-authors-dir"):
        stream_semantic_scholar_authors(str(empty), str(tmp_path / "out.parquet"))


def _homepage_record(
    pid: str, aliases: list[str], affiliations: list[str] | None = None
) -> dict:
    return {
        "pid": pid,
        "crossref": None,
        "aliases": aliases,
        "affiliations": affiliations or [],
        "urls": [],
    }


def test_enrich_folds_match():
    homepages = {"12/34": _homepage_record("12/34", ["Kun Zhang 0006"])}
    s2_authors = [
        {
            "authorid": "143972713",
            "name": "Kun Zhang",
            "aliases": [],
            "affiliations": ["Carnegie Mellon University"],
            "homepage": "",
            "papercount": 24,
            "citationcount": 122,
            "hindex": 7,
            "dblp_names": ["Kun Zhang 0006", "Kun Zhang 0016"],
            "orcid": [],
        }
    ]

    enriched = enrich_homepages_with_semantic_scholar(homepages, s2_authors)

    record = enriched["12/34"]
    assert record["aliases"] == ["Kun Zhang", "Kun Zhang 0006"]
    assert record["affiliations"] == ["Carnegie Mellon University"]


def test_enrich_no_match():
    homepages = {"12/34": _homepage_record("12/34", ["Jane Doe"])}
    s2_authors = [
        {
            "authorid": "1",
            "name": "Unrelated Person",
            "aliases": [],
            "affiliations": [],
            "homepage": "",
            "papercount": 0,
            "citationcount": 0,
            "hindex": 0,
            "dblp_names": ["Someone Else"],
            "orcid": [],
        }
    ]

    enriched = enrich_homepages_with_semantic_scholar(homepages, s2_authors)
    assert enriched["12/34"]["aliases"] == ["Jane Doe"]


def test_enrich_skips_unmapped():
    homepages = {"12/34": _homepage_record("12/34", ["Jane Doe"])}
    s2_authors = [
        {
            "authorid": "1",
            "name": "Jane Doe",
            "aliases": [],
            "affiliations": ["Somewhere"],
            "homepage": "",
            "papercount": 0,
            "citationcount": 0,
            "hindex": 0,
            "dblp_names": [],
            "orcid": [],
        }
    ]

    enriched = enrich_homepages_with_semantic_scholar(homepages, s2_authors)
    # No DBLP mapping on the S2 side, so nothing to corroborate against
    assert enriched["12/34"]["affiliations"] == []


def _authors_cache(tmp_path) -> str:
    source_dir = _write_part(tmp_path, "part-01.jsonl.gz", [RAW_KUN_ZHANG])
    parquet_path = str(tmp_path / "s2_authors.parquet")
    stream_semantic_scholar_authors(str(source_dir), parquet_path)
    return parquet_path


def _homepages_cache(tmp_path) -> str:
    path = str(tmp_path / "dblp.homepages.parquet")
    write_homepages({"12/34": _homepage_record("12/34", ["Kun Zhang 0006"])}, path)
    return path


def test_enriched_written_separately(tmp_path):
    """Separate, not in place: dropping --s2-authors-dir later must not keep
    silently using S2-derived aliases."""
    homepages = _homepages_cache(tmp_path)
    authors = _authors_cache(tmp_path)

    enriched = build_enriched_homepages(homepages, authors)

    assert enriched == str(tmp_path / "dblp.homepages.enriched.parquet")
    assert "Kun Zhang" in load_homepages(enriched)["12/34"]["aliases"]
    # The unenriched index is untouched
    assert load_homepages(homepages)["12/34"]["aliases"] == ["Kun Zhang 0006"]


def test_enriched_skips_when_fresh(tmp_path, caplog):
    """The pass streams 100M+ author rows; it must not rerun for nothing."""
    homepages = _homepages_cache(tmp_path)
    authors = _authors_cache(tmp_path)
    build_enriched_homepages(homepages, authors)

    with caplog.at_level(logging.INFO):
        build_enriched_homepages(homepages, authors)

    assert "skipping S2 enrichment" in caplog.text


def test_enriched_reruns_on_new_authors(tmp_path):
    homepages = _homepages_cache(tmp_path)
    authors = _authors_cache(tmp_path)
    enriched = build_enriched_homepages(homepages, authors)
    assert "Kun Zhang" in load_homepages(enriched)["12/34"]["aliases"]

    # A rebuilt authors cache with a different author for the same DBLP name
    source_dir = tmp_path / "authors"
    with gzip.open(source_dir / "part-01.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(
            json.dumps({**RAW_KUN_ZHANG, "name": "K. Zhang", "aliases": ["Zhang, Kun"]})
            + "\n"
        )
    stream_semantic_scholar_authors(str(source_dir), authors, force=True)

    build_enriched_homepages(homepages, authors)
    assert "Zhang, Kun" in load_homepages(enriched)["12/34"]["aliases"]
