from __future__ import annotations

import pyarrow.parquet as pq
import pytest
from lxml import etree

from coi_dams import cache
from coi_dams.dblp import (
    _parse_doi,
    _parse_homepage,
    _parse_publication,
    _parse_year,
    fold_crossref_aliases,
    homepages_path,
    load_homepages,
    stream_dblp,
    write_homepages,
)


def element(xml: str) -> etree._Element:
    return etree.fromstring(xml)


def test_publication_fields():
    parsed = _parse_publication(
        element("""
        <inproceedings key="conf/x/1">
          <author>Jane Doe</author>
          <author>John Roe</author>
          <title>A Paper</title>
          <year>2020</year>
          <booktitle>SIGMOD</booktitle>
          <ee>https://doi.org/10.1/abc</ee>
        </inproceedings>
    """),
        min_year=2016,
    )
    assert parsed == {
        "key": "conf/x/1",
        "title": "A Paper",
        "authors": ["Jane Doe", "John Roe"],
        "year": 2020,
        "venue": "SIGMOD",
        "pub_type": "inproceedings",
        "doi": "https://doi.org/10.1/abc",
    }


def test_publication_cutoff():
    old = element("<article key='k'><year>1999</year><title>t</title></article>")
    assert _parse_publication(old, min_year=2016) is None


def test_publication_journal_over_booktitle():
    parsed = _parse_publication(
        element("""
        <article key="k"><year>2020</year><journal>PVLDB</journal><booktitle>Ignored</booktitle></article>
    """),
        min_year=2016,
    )
    assert parsed["venue"] == "PVLDB"


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        ("<article><year>2020</year></article>", 2020),
        ("<article><year>not a year</year></article>", 0),
        ("<article/>", 0),
    ],
)
def test_parse_year_never_raises(xml, expected):
    assert _parse_year(element(xml)) == expected


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        ("<a><ee>https://doi.org/10.1/x</ee></a>", "https://doi.org/10.1/x"),
        (
            "<a><ee>https://example.org/pdf</ee><ee>https://doi.org/10.1/x</ee></a>",
            "https://doi.org/10.1/x",
        ),
        ("<a><doi> 10.1/x </doi></a>", "https://doi.org/10.1/x"),
        ("<a><ee>https://example.org/pdf</ee></a>", ""),
        ("<a/>", ""),
    ],
)
def test_doi_prefers_link(xml, expected):
    assert _parse_doi(element(xml)) == expected


def test_homepage_fields():
    parsed = _parse_homepage(element("""
        <www key="homepages/12/3456">
          <author>Jane Doe</author>
          <author>Jane A. Doe</author>
          <note type="affiliation">University of California, Berkeley</note>
          <note type="award">Ignored</note>
          <url>https://example.org/jane</url>
        </www>
    """))
    assert parsed["pid"] == "12/3456"
    assert parsed["crossref"] is None
    assert parsed["aliases"] == ["Jane Doe", "Jane A. Doe"]
    assert parsed["affiliations"] == ["University of California, Berkeley"]
    assert parsed["urls"] == ["https://example.org/jane"]


def test_homepage_crossref_names():
    """Pointer records are kept so fold_crossref_aliases has something to fold."""
    parsed = _parse_homepage(element("""
        <www key="homepages/p/Pointer">
          <author>J. Doe</author>
          <crossref>homepages/12/3456</crossref>
        </www>
    """))
    assert parsed["crossref"] == "homepages/12/3456"
    assert parsed["aliases"] == ["J. Doe"]


def test_homepage_ignores_other_www():
    assert (
        _parse_homepage(element("<www key='conf/x/1'><title>t</title></www>")) is None
    )


def test_fold_crossref():
    """A secondary pid of a merged profile stays resolvable."""
    raw = {
        "t/Target": {
            "pid": "t/Target",
            "crossref": None,
            "aliases": ["Jane Doe"],
            "affiliations": [],
            "urls": [],
        },
        "p/Pointer": {
            "pid": "p/Pointer",
            "crossref": "homepages/t/Target",
            "aliases": ["J. Doe"],
            "affiliations": [],
            "urls": [],
        },
    }
    folded = fold_crossref_aliases(raw)
    assert folded["t/Target"]["aliases"] == ["J. Doe", "Jane Doe"]  # Sorted union
    assert folded["p/Pointer"]["crossref"] == "homepages/t/Target"


def test_fold_crossref_dangling():
    raw = {
        "p": {
            "pid": "p",
            "crossref": "homepages/gone",
            "aliases": [],
            "affiliations": [],
            "urls": [],
        }
    }
    assert fold_crossref_aliases(raw) == raw


DUMP = """<?xml version="1.0" encoding="UTF-8"?>
<dblp>
  <www key="homepages/t/Target"><author>Jane Doe</author><url>https://example.org/j</url></www>
  <www key="homepages/p/Pointer"><author>J. Doe</author><crossref>homepages/t/Target</crossref></www>
  <inproceedings key="conf/x/1">
    <author>Jane Doe</author><author>John Roe</author>
    <title>Recent</title><year>2020</year><booktitle>SIGMOD</booktitle>
  </inproceedings>
  <article key="journals/y/2">
    <author>Jane Doe</author><title>Too old</title><year>1999</year><journal>TODS</journal>
  </article>
</dblp>
"""


def test_stream_dblp_writes_both(tmp_path):
    import pyarrow.parquet as pq

    xml = tmp_path / "dblp.xml"
    xml.write_text(DUMP, encoding="utf-8")
    (tmp_path / "dblp.dtd").write_text("", encoding="utf-8")
    parquet = str(tmp_path / "dblp.parquet")

    stream_dblp(str(xml), parquet, min_year=2016)

    rows = pq.read_table(parquet).to_pylist()
    assert [row["key"] for row in rows] == ["conf/x/1"]  # The 1999 article is cut
    assert rows[0]["authors"] == ["Jane Doe", "John Roe"]

    homepages = load_homepages(homepages_path(parquet))
    assert homepages["t/Target"]["aliases"] == ["J. Doe", "Jane Doe"]
    assert homepages["p/Pointer"]["crossref"] == "homepages/t/Target"


def test_stream_dblp_failure_no_partial(tmp_path):
    """A truncated Parquet with a fresh mtime is not treated as up to date."""
    xml = tmp_path / "dblp.xml"
    xml.write_text("<dblp><inproceedings key='k'><year>2020</year>", encoding="utf-8")
    (tmp_path / "dblp.dtd").write_text("", encoding="utf-8")
    parquet = tmp_path / "dblp.parquet"

    with pytest.raises(etree.XMLSyntaxError):
        stream_dblp(str(xml), str(parquet), min_year=2016)

    assert not parquet.exists()
    assert not (tmp_path / "dblp.parquet.partial").exists()
    assert not (tmp_path / "dblp.homepages.parquet.partial").exists()


def _parsed_dump(tmp_path, min_year: int = 2016) -> str:
    """A parsed dump; returns the publications cache path."""
    xml = tmp_path / "dblp.xml"
    xml.write_text(DUMP, encoding="utf-8")
    (tmp_path / "dblp.dtd").write_text("", encoding="utf-8")
    parquet = str(tmp_path / "dblp.parquet")
    stream_dblp(str(xml), parquet, min_year=min_year)
    return parquet


def test_stream_dblp_skips_when_fresh(tmp_path, caplog):
    import logging

    parquet = _parsed_dump(tmp_path)
    xml = str(tmp_path / "dblp.xml")

    # Plant a marker in the homepage index, a skipped parse leaves it there
    marker = {
        "sentinel": {
            "pid": "sentinel",
            "crossref": None,
            "aliases": [],
            "affiliations": [],
            "urls": [],
        }
    }
    write_homepages(
        marker, homepages_path(parquet), manifest=cache.read_manifest(parquet)
    )

    with caplog.at_level(logging.INFO):
        stream_dblp(xml, parquet, min_year=2016)
    assert "skipping parse" in caplog.text
    assert load_homepages(homepages_path(parquet)) == marker

    stream_dblp(xml, parquet, min_year=2016, force=True)
    assert "sentinel" not in load_homepages(homepages_path(parquet))


def test_stream_dblp_cutoff_mismatch(tmp_path):
    """The bug this exists for: an mtime check called the 2016 cache fresh."""
    parquet = _parsed_dump(tmp_path, min_year=2016)

    with pytest.raises(SystemExit) as error:
        stream_dblp(str(tmp_path / "dblp.xml"), parquet, min_year=2010)

    message = str(error.value)
    assert cache.PARAM_CUTOFF_YEAR in message
    assert "2016" in message and "2010" in message
    # The message has to say how to recover
    assert "--force" in message


def test_force_rebuilds_new_cutoff(tmp_path):
    parquet = _parsed_dump(tmp_path, min_year=2016)
    xml = str(tmp_path / "dblp.xml")

    stream_dblp(xml, parquet, min_year=1990, force=True)

    manifest = cache.read_manifest(parquet)
    assert manifest is not None
    assert manifest.params == {cache.PARAM_CUTOFF_YEAR: 1990}
    # Rebuilt, the older article is now in range.
    assert len(pq.read_table(parquet).to_pylist()) == 2

    # The rebuilt cache is reusable at the new setting without --force
    stream_dblp(xml, parquet, min_year=1990)


def test_no_manifest_is_refused(tmp_path):
    """A pre-manifest artefact cannot vouch for the cutoff it was built under."""
    parquet = _parsed_dump(tmp_path, min_year=2016)
    table = pq.read_table(parquet)
    pq.write_table(table.replace_schema_metadata({}), parquet)

    with pytest.raises(SystemExit, match="no readable cache manifest"):
        stream_dblp(str(tmp_path / "dblp.xml"), parquet, min_year=2016)


def test_stream_dblp_missing_dtd(tmp_path):
    xml = tmp_path / "dblp.xml"
    xml.write_text(DUMP, encoding="utf-8")
    with pytest.raises(SystemExit, match=r"dblp\.dtd not found"):
        stream_dblp(str(xml), str(tmp_path / "dblp.parquet"), min_year=2016)


def test_stream_dblp_compressed_dump(tmp_path):
    """The parser reads an extracted dump; the fetch script does the unpacking."""
    import gzip

    archive = tmp_path / "dblp.xml.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as fh:
        fh.write(DUMP)

    with pytest.raises(SystemExit) as error:
        stream_dblp(str(archive), str(tmp_path / "out.parquet"), min_year=2016)

    message = str(error.value)
    assert "extracted dump" in message
    assert "fetch_data_sources.py" in message
