from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coi_dams.coi import check_unique_pairs, coi_analysis, resolve_emails
from coi_dams.constants import DBLP_SCHEMA, DEFAULT_WEIGHTS
from coi_dams.dblp import write_homepages
from coi_dams.evidence import format_paths
from coi_dams.mappings import (
    build_alias_node_df,
    build_alias_pid_df,
    build_email_mapping_df,
    build_node_mapping_df,
)
from conftest import requires_java

pytestmark = requires_java

PAPERS = [
    ("p1", ["A", "B"]),
    ("p2", ["B", "C"]),
    ("p3", ["A", "B"]),
    ("p4", ["D", "E"]),
]

HOMEPAGES = {
    name.lower(): {
        "pid": name.lower(),
        "crossref": None,
        "aliases": [name],
        "affiliations": [],
        "urls": [],
    }
    for name in "ABCDE"
}

# Everyone is a known user, so nothing is dropped for being unresolvable
EMAIL_INDEX = {
    name: (f"{name.lower()}@example.org", name.lower(), "certain") for name in "ABCDE"
}


@pytest.fixture(scope="module")
def homepages_parquet(tmp_path_factory):
    """HOMEPAGES as the on-disk index build_alias_pid_df now streams."""
    path = str(tmp_path_factory.mktemp("index") / "dblp.homepages.parquet")
    write_homepages(HOMEPAGES, path)
    return path


def write_papers(papers, path):
    rows = [
        {
            "key": key,
            "title": key.upper(),
            "authors": authors,
            "year": 2020,
            "venue": "SIGMOD",
            "pub_type": "inproceedings",
            "doi": "",
        }
        for key, authors in papers
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=DBLP_SCHEMA), path)
    return path


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory):
    return write_papers(PAPERS, str(tmp_path_factory.mktemp("data") / "pubs.parquet"))


def run_pipeline(
    spark,
    parquet_path,
    homepages_parquet,
    checkpoint_dir,
    email_index=EMAIL_INDEX,
    **analysis_kwargs,
):
    """The whole chain, returning {(email_a, email_b): row} keyed on the pair."""
    alias_pid_df = build_alias_pid_df(spark, homepages_parquet)
    email_mapping = build_email_mapping_df(spark, email_index)
    mapping = build_node_mapping_df(email_mapping)
    with coi_analysis(
        spark,
        parquet_path,
        DEFAULT_WEIGHTS,
        checkpoint_dir=str(checkpoint_dir),
        source_authors=mapping.select("node_id"),
        alias_node_df=build_alias_node_df(alias_pid_df, email_mapping),
        **analysis_kwargs,
    ) as df_coi:
        pairs = resolve_emails(df_coi, mapping)
        check_unique_pairs(pairs)
        rows = format_paths(pairs, alias_pid_df).collect()
    return {(row["email_a"], row["email_b"]): row for row in rows}


def email(name: str) -> str:
    return f"{name.lower()}@example.org"


def run_graph(spark, tmp_path, papers, aliases=None, max_depth=3, users=None):
    """run_pipeline on its own papers. `aliases` maps a pid to its author names.

    `users` limits the known users to those pids, by default all of them.
    """
    aliases = dict(aliases or {})
    named = {name for names in aliases.values() for name in names}
    for _, authors in papers:
        aliases.update({a.lower(): [a] for a in authors if a not in named})
    homepages = {
        pid: {**HOMEPAGES["a"], "pid": pid, "aliases": names}
        for pid, names in aliases.items()
    }
    index = {
        name: (f"{pid}@example.org", pid, "certain")
        for pid, names in aliases.items()
        for name in names
        if users is None or pid in users
    }
    homepages_path = str(tmp_path / "homepages.parquet")
    write_homepages(homepages, homepages_path)
    return run_pipeline(
        spark,
        write_papers(papers, str(tmp_path / "pubs.parquet")),
        homepages_path,
        tmp_path / "ckpt",
        email_index=index,
        max_depth=max_depth,
    )


def test_direct_and_two_hop_pairs(spark, parquet_path, homepages_parquet, tmp_path):
    result = run_pipeline(
        spark, parquet_path, homepages_parquet, tmp_path / "ckpt", max_depth=2
    )

    assert set(result) == {
        (email("A"), email("B")),
        (email("B"), email("C")),
        (email("A"), email("C")),
        (email("D"), email("E")),
    }

    direct = result[(email("A"), email("B"))]
    assert direct["depth"] == 1
    assert direct["coi_type"] == "direct_coauthor"
    # A and B share two papers (p1, p3)
    assert direct["evidence_path_count"] == 2
    assert direct["coi_score"] == 2 * DEFAULT_WEIGHTS["depth_1"]

    submarine = result[(email("A"), email("C"))]
    assert submarine["depth"] == 2
    assert submarine["coi_type"] == "indirect_depth_2"
    assert (
        submarine["coi_score"]
        == submarine["evidence_path_count"] * DEFAULT_WEIGHTS["depth_2"]
    )


def test_depth_one_alternate_paths(spark, parquet_path, homepages_parquet, tmp_path):
    result = run_pipeline(
        spark, parquet_path, homepages_parquet, tmp_path / "ckpt", max_depth=2
    )

    # Both shared papers are recorded
    direct = result[(email("A"), email("B"))]
    assert direct["evidence_path_count"] == 2
    assert set(direct["evidence_papers"].split(";")) == {
        "Conference(A/a, B/b, p1)",
        "Conference(A/a, B/b, p3)",
    }

    # A and C are linked by one chain of people, A-B-C
    assert result[(email("A"), email("C"))]["evidence_path_count"] == 1


def test_two_hop_paths_count_chains_of_people(spark, tmp_path):
    papers = [
        (f"p{i}", pair)
        for i, pair in enumerate(
            [
                ["A", "X1"],
                ["X1", "B"],
                ["A", "X2"],
                ["X2", "B"],
                ["A", "X3"],
                ["X3", "B"],
            ]
        )
    ]
    result = run_graph(spark, tmp_path, papers, max_depth=2)

    pair = result[(email("A"), email("B"))]
    assert pair["depth"] == 2
    assert pair["evidence_path_count"] == 3
    assert pair["coi_score"] == 3 * DEFAULT_WEIGHTS["depth_2"]


def test_papers_on_a_hop_are_not_counted_beyond_depth_one(spark, tmp_path):
    papers = [(f"ab{i}", ["A", "B"]) for i in range(2)]
    papers += [(f"bc{i}", ["B", "C"]) for i in range(3)]
    result = run_graph(spark, tmp_path, papers, max_depth=2)

    assert result[(email("A"), email("B"))]["evidence_path_count"] == 2
    two_hop = result[(email("A"), email("C"))]
    assert two_hop["evidence_path_count"] == 1
    assert two_hop["coi_score"] == DEFAULT_WEIGHTS["depth_2"]


def test_aliases_of_one_person_are_one_path(spark, tmp_path):
    papers = [
        ("p1", ["A", "B1"]),
        ("p2", ["B1", "C"]),
        ("p3", ["A", "B2"]),
        ("p4", ["B2", "C"]),
    ]
    result = run_graph(spark, tmp_path, papers, aliases={"b": ["B1", "B2"]})

    assert result[(email("A"), email("C"))]["evidence_path_count"] == 1


def test_ambiguous_alias_of_a_user_is_the_user(spark, tmp_path):
    # B is an alias of pid b2 too, but it is also a name of the user b
    papers = [("p1", ["A", "B"]), ("p2", ["A", "B1"])]
    aliases = {"b": ["B1", "B"], "b2": ["B"]}
    result = run_graph(spark, tmp_path, papers, aliases, users={"a", "b"})

    assert set(result) == {(email("A"), email("b"))}
    pair = result[(email("A"), email("b"))]
    assert pair["depth"] == 1
    assert pair["evidence_path_count"] == 2
    assert pair["coi_score"] == 2 * DEFAULT_WEIGHTS["depth_1"]


def test_ambiguous_alias_of_a_user_in_the_middle(spark, tmp_path):
    papers = [
        ("p1", ["A", "B1"]),
        ("p2", ["B1", "C"]),
        ("p3", ["A", "B"]),
        ("p4", ["B", "C"]),
    ]
    aliases = {"b": ["B1", "B"], "b2": ["B"]}
    result = run_graph(spark, tmp_path, papers, aliases, users={"a", "b", "c"})

    assert set(result) == {
        (email("A"), email("b")),
        (email("b"), email("C")),
        (email("A"), email("C")),
    }
    pair = result[(email("A"), email("C"))]
    assert pair["depth"] == 2
    assert pair["evidence_path_count"] == 1
    assert pair["coi_score"] == DEFAULT_WEIGHTS["depth_2"]


def test_alias_missing_from_the_email_index_follows_the_pid(
    spark, parquet_path, homepages_parquet, tmp_path
):
    # B is not listed for its user, but it maps to the pid of one
    index = {name: v for name, v in EMAIL_INDEX.items() if name != "B"}
    index["B2"] = (email("B"), "b", "certain")
    result = run_pipeline(
        spark,
        parquet_path,
        homepages_parquet,
        tmp_path / "ckpt",
        email_index=index,
        max_depth=2,
    )

    assert result[(email("A"), email("B"))]["evidence_path_count"] == 2
    assert result[(email("A"), email("C"))]["depth"] == 2


def test_duplicate_email_pairs_are_rejected(spark):
    columns = ["email_a", "email_b"]
    check_unique_pairs(spark.createDataFrame([("a", "b"), ("a", "c")], columns))
    # Swapped columns are the same pair
    duplicated = spark.createDataFrame([("a", "b"), ("b", "a")], columns)
    with pytest.raises(RuntimeError, match="1 email pair"):
        check_unique_pairs(duplicated)


def test_depth_three_is_scored_without_evidence(spark, tmp_path):
    papers = [("p1", ["A", "X"]), ("p2", ["X", "Y"]), ("p3", ["Y", "B"])]
    result = run_graph(spark, tmp_path, papers)

    pair = result[(email("A"), email("B"))]
    assert pair["depth"] == 3
    assert pair["evidence_path_count"] == 1
    assert pair["coi_score"] == DEFAULT_WEIGHTS["depth_3"]
    assert pair["evidence_papers"] is None


def test_two_hop_evidence_direction(spark, parquet_path, homepages_parquet, tmp_path):
    result = run_pipeline(
        spark, parquet_path, homepages_parquet, tmp_path / "ckpt", max_depth=2
    )

    hops = result[(email("A"), email("C"))]["evidence_papers"].split("+")
    assert hops[0].startswith("Conference(A/a, B/b, ")
    assert hops[1].startswith("Conference(B/b, C/c, ")


def test_hub_cap_counts_coauthors(spark, parquet_path, homepages_parquet, tmp_path):
    """Counting edge rows would charge A twice for its two papers with B.

    B has two distinct co-authors (A and C) and is capped at 1. A has one
    distinct co-author across two papers and must not be.
    """
    result = run_pipeline(
        spark,
        parquet_path,
        homepages_parquet,
        tmp_path / "ckpt",
        max_depth=2,
        max_hub_degree=1,
    )
    # B is excluded as a pivot, so the two-hop pair through it is not found.
    assert (email("A"), email("C")) not in result
    # Every direct pair, including A's, survives untouched
    assert (email("A"), email("B")) in result
    assert result[(email("A"), email("B"))]["evidence_path_count"] == 2


def test_bucketing_same_result(spark, parquet_path, homepages_parquet, tmp_path):
    """With one bucket the hop's cache is not released before it is used."""
    single = run_pipeline(
        spark,
        parquet_path,
        homepages_parquet,
        tmp_path / "ckpt1",
        max_depth=2,
        source_buckets=1,
    )
    bucketed = run_pipeline(
        spark,
        parquet_path,
        homepages_parquet,
        tmp_path / "ckpt4",
        max_depth=2,
        source_buckets=4,
    )

    assert set(single) == set(bucketed)
    for key, row in single.items():
        assert bucketed[key]["depth"] == row["depth"]
        assert bucketed[key]["evidence_path_count"] == row["evidence_path_count"]
