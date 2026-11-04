from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coi_dams.constants import DBLP_SCHEMA
from coi_dams.dblp import write_homepages
from coi_dams.qualify import (
    Paper,
    PublicationIndex,
    _env_venues,
    _papers_per_venue,
    _total_publications,
    _venue_label,
    build_publication_index,
    check_reviewer_qualifications,
)


@pytest.mark.parametrize(
    ("venue", "expected"),
    [
        ("SIGMOD Conference", "SIGMOD"),
        ("Proceedings of the ACM on Management of Data", "SIGMOD"),
        ("PVLDB", "VLDB"),
        ("ICDE", "ICDE"),
        ("IEEE Trans. Knowl. Data Eng.", None),
        ("", None),
    ],
)
def test_venue_label(venue, expected):
    assert _venue_label(venue) == expected


def index(qualifying=None, identities=None) -> PublicationIndex:
    return PublicationIndex(qualifying or {}, identities or {})


def test_papers_per_venue_dedupe():
    paper = Paper("A Paper", "https://doi.org/10.1/x", "SIGMOD")
    # The same work reached under two spellings of one name
    result = _papers_per_venue(
        {"Jane Doe", "J. Doe"},
        index(
            qualifying={
                "Jane Doe": [paper],
                "J. Doe": [Paper("a paper", paper.doi, "SIGMOD")],
            }
        ),
    )
    assert len(result["SIGMOD"]) == 1
    assert result["VLDB"] == []


def test_total_matches_venue_identity():
    """Both counts in a report row are computed the same way."""
    identities = {
        "Jane Doe": {("a paper", "doi:1"), ("another", "")},
        "J. Doe": {("a paper", "doi:1")},
    }
    assert (
        _total_publications({"Jane Doe", "J. Doe"}, index(identities=identities)) == 2
    )


def write_parquet(tmp_path, rows):
    table = pa.Table.from_pylist(rows, schema=DBLP_SCHEMA)
    path = str(tmp_path / "pubs.parquet")
    pq.write_table(table, path)
    return path


def publication(key, authors, venue, title="T", doi=""):
    return {
        "key": key,
        "title": title,
        "authors": authors,
        "year": 2020,
        "venue": venue,
        "pub_type": "inproceedings",
        "doi": doi,
    }


def test_publication_index_filters_aliases(tmp_path):
    path = write_parquet(
        tmp_path,
        [
            publication(
                "k1", ["Jane Doe", "Bystander"], "SIGMOD Conference", title="P1"
            ),
            publication("k2", ["Jane Doe"], "Some Workshop", title="P2"),
            publication("k3", ["Nobody Relevant"], "PVLDB", title="P3"),
        ],
    )

    result = build_publication_index(path, {"Jane Doe"})

    assert set(result.paper_identities) == {"Jane Doe"}
    assert result.paper_identities["Jane Doe"] == {("p1", ""), ("p2", "")}
    assert [paper.title for paper in result.qualifying_papers["Jane Doe"]] == ["P1"]
    assert "Bystander" not in result.qualifying_papers


_DEFAULT_VENUES = {"FALLBACK": ["fallback"]}


def test_env_venues_per_line(monkeypatch):
    # Blank lines are ignored
    monkeypatch.setenv(
        "COI_TEST_VENUES",
        "\n  SIGMOD: sigmod, management of data\n\n  VLDB: vldb, pvldb\n",
    )
    assert _env_venues("COI_TEST_VENUES", _DEFAULT_VENUES) == {
        "SIGMOD": ["sigmod", "management of data"],
        "VLDB": ["vldb", "pvldb"],
    }


@pytest.mark.parametrize("raw", ["SIGMOD", "SIGMOD:", ": sigmod", "SIGMOD sigmod"])
def test_env_venues_bad_entry(monkeypatch, raw):
    monkeypatch.setenv("COI_TEST_VENUES", raw)
    with pytest.raises(SystemExit) as error:
        _env_venues("COI_TEST_VENUES", _DEFAULT_VENUES)
    assert "COI_TEST_VENUES" in str(error.value)


def write_users_csv(tmp_path, *rows):
    header = "#First Name\tLast Name\tEmail\tRoles\tDBLP URL"
    lines = [header] + [
        "\t".join([first, last, email, "reviewer", dblp_url])
        for first, last, email, dblp_url in rows
    ]
    path = tmp_path / "Users.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def homepage(pid, aliases, crossref=None):
    return {
        "pid": pid,
        "crossref": crossref,
        "aliases": list(aliases),
        "affiliations": [],
        "urls": [],
    }


def test_qualifications_exclude_failed(tmp_path, monkeypatch):
    monkeypatch.setattr("coi_dams.qualify.REVIEWER_MIN_PAPERS", 1)

    # "Jane Doe" resolves via her DBLP pid, "John Roe" shares his name with two entities
    pubs_path = write_parquet(
        tmp_path,
        [
            publication("k1", ["Jane Doe"], "SIGMOD Conference", title="P1"),
            publication("k2", ["John Roe"], "SIGMOD Conference", title="P2"),
        ],
    )
    homepages_path_ = str(tmp_path / "pubs.homepages.parquet")
    write_homepages(
        {
            "a": homepage("a", ["Jane Doe"]),
            "b": homepage("b", ["John Roe"]),
            "c": homepage("c", ["John Roe"]),
        },
        homepages_path_,
    )
    users_path = write_users_csv(
        tmp_path,
        ("Jane", "Doe", "jane@example.org", "https://dblp.org/pid/a"),
        ("John", "Roe", "john@example.org", ""),
    )
    output_path = str(tmp_path / "qualified.csv")
    unresolved_path = str(tmp_path / "unresolved.csv")

    check_reviewer_qualifications(
        pubs_path,
        users_path,
        output_path,
        homepages_parquet=homepages_path_,
        unresolved_output_path=unresolved_path,
    )

    with open(output_path, encoding="utf-8") as fh:
        report = fh.read()
    assert "jane@example.org" in report
    assert "john@example.org" not in report

    with open(unresolved_path, encoding="utf-8") as fh:
        unresolved = fh.read()
    assert "john@example.org" in unresolved
    assert "jane@example.org" not in unresolved
