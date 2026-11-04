from __future__ import annotations

import gzip
import json
import logging

import pytest

from coi_dams.semantic_scholar_papers import (
    parse_paper,
    stream_semantic_scholar_papers,
    synthetic_author_id,
)


def _write_part(tmp_path, name: str, records: list[dict]):
    source_dir = tmp_path / "papers"
    source_dir.mkdir(exist_ok=True)
    with gzip.open(source_dir / name, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return source_dir


RAW_PAPER = {
    "corpusid": 213579185,
    "externalids": {"DOI": "10.1002/app.48759", "DBLP": None},
    "url": "https://www.semanticscholar.org/paper/x",
    "title": "Electrospinning of circumferentially aligned polymer nanofibers",
    "authors": [
        {"authorId": "2261072084", "name": "Shichen Li"},
        {"authorId": "14445644", "name": "Bong-Kee Lee"},
    ],
    "venue": "",
    "publicationvenueid": None,
    "year": 2020,
    "referencecount": 25,
    "citationcount": 16,
    "influentialcitationcount": 0,
    "isopenaccess": False,
    "s2fieldsofstudy": None,
    "publicationtypes": ["JournalArticle"],
    "publicationdate": "2020-06-10",
    "journal": {
        "name": "Journal of Applied Polymer Science",
        "pages": "48759",
        "volume": "137",
    },
}

OLD_PAPER = {**RAW_PAPER, "corpusid": 1, "year": 1999}

SOLO_AUTHOR_PAPER = {
    **RAW_PAPER,
    "corpusid": 2,
    "authors": [{"authorId": "1", "name": "Solo Author"}],
}


def test_paper_mapped_author():
    name_map = {"2261072084": ["Shi-Chen Li 0001"]}
    parsed = parse_paper(RAW_PAPER, name_map, min_year=2016)
    assert parsed["authors"][0] == "Shi-Chen Li 0001"
    assert parsed["authors"][1] == synthetic_author_id("14445644", "Bong-Kee Lee")


def test_paper_synthetic_identity():
    parsed = parse_paper(RAW_PAPER, {}, min_year=2016)
    assert parsed["authors"] == [
        synthetic_author_id("2261072084", "Shichen Li"),
        synthetic_author_id("14445644", "Bong-Kee Lee"),
    ]


def test_paper_fields():
    parsed = parse_paper(RAW_PAPER, {}, min_year=2016)
    assert parsed["key"] == "s2:213579185"
    assert (
        parsed["venue"] == "Journal of Applied Polymer Science"
    )  # Falls back to journal.name
    assert parsed["doi"] == "https://doi.org/10.1002/app.48759"
    assert parsed["pub_type"] == "article"
    assert parsed["year"] == 2020


def test_paper_cutoff():
    assert parse_paper(OLD_PAPER, {}, min_year=2016) is None


def test_paper_needs_two_authors():
    assert parse_paper(SOLO_AUTHOR_PAPER, {}, min_year=2016) is None


@pytest.mark.parametrize(
    ("publicationtypes", "expected"),
    [
        (["Conference"], "inproceedings"),
        (["Book"], "book"),
        (["BookSection"], "incollection"),
        (["JournalArticle"], "article"),
        (["CaseReport"], "article"),  # Unmapped type falls back
        (None, "article"),
        ([], "article"),
    ],
)
def test_paper_pub_types(publicationtypes, expected):
    raw = {**RAW_PAPER, "publicationtypes": publicationtypes}
    assert parse_paper(raw, {}, min_year=2016)["pub_type"] == expected


def test_s2_papers_skips_when_fresh(tmp_path, caplog):
    source_dir = _write_part(tmp_path, "part-01.jsonl.gz", [RAW_PAPER])
    parquet_path = str(tmp_path / "s2_papers.parquet")
    stream_semantic_scholar_papers(str(source_dir), parquet_path, dict, min_year=2016)

    with caplog.at_level(logging.INFO):
        stream_semantic_scholar_papers(
            str(source_dir), parquet_path, dict, min_year=2016
        )
    assert "skipping parse" in caplog.text


def test_s2_papers_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="--s2-papers-dir"):
        stream_semantic_scholar_papers(
            str(empty), str(tmp_path / "out.parquet"), dict, min_year=2016
        )


def test_s2_papers_fresh_does_not_build_name_map(tmp_path):
    source_dir = _write_part(tmp_path, "part-01.jsonl.gz", [RAW_PAPER])
    parquet_path = str(tmp_path / "s2_papers.parquet")
    stream_semantic_scholar_papers(str(source_dir), parquet_path, dict, min_year=2016)

    def fail():
        raise AssertionError("name map built despite fresh cache")

    stream_semantic_scholar_papers(str(source_dir), parquet_path, fail, min_year=2016)
