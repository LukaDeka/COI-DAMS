from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coi_dams.constants import DBLP_SCHEMA
from coi_dams.publications import is_combined_cache_stale, merge_and_dedup_parquets


def _write_parquet(path: str, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=DBLP_SCHEMA), path)


def _row(key: str, doi: str = "") -> dict:
    return {
        "key": key,
        "title": key,
        "authors": ["Jane Doe", "John Roe"],
        "year": 2020,
        "venue": "SIGMOD",
        "pub_type": "inproceedings",
        "doi": doi,
    }


def test_merge_drops_duplicate_doi(tmp_path):
    dblp_path = str(tmp_path / "dblp.parquet")
    s2_path = str(tmp_path / "s2.parquet")
    combined_path = str(tmp_path / "combined.parquet")
    shared_doi = "https://doi.org/10.1/x"
    _write_parquet(dblp_path, [_row("conf/x/1", doi=shared_doi)])
    _write_parquet(s2_path, [_row("s2:1", doi=shared_doi), _row("s2:2")])

    merge_and_dedup_parquets([dblp_path, s2_path], combined_path)

    keys = sorted(row["key"] for row in pq.read_table(combined_path).to_pylist())
    # DBLP's row wins the DOI collision, s2:1 is dropped and s2:2 has no DOI so it is kept
    assert keys == ["conf/x/1", "s2:2"]


def test_merge_keeps_no_doi_rows(tmp_path):
    dblp_path = str(tmp_path / "dblp.parquet")
    combined_path = str(tmp_path / "combined.parquet")
    _write_parquet(dblp_path, [_row("conf/x/1"), _row("conf/x/2")])

    merge_and_dedup_parquets([dblp_path], combined_path)

    assert pq.read_table(combined_path).num_rows == 2


def test_merge_failure_leaves_no_partial(tmp_path):
    combined_path = str(tmp_path / "combined.parquet")
    # The point is the .partial cleanup, not the exception type
    with pytest.raises(OSError):
        merge_and_dedup_parquets([str(tmp_path / "missing.parquet")], combined_path)
    assert not (tmp_path / "combined.parquet.partial").exists()


def test_fresh_merge_not_stale(tmp_path):
    source = str(tmp_path / "source.parquet")
    combined = str(tmp_path / "combined.parquet")
    _write_parquet(source, [_row("conf/x/1")])

    merge_and_dedup_parquets([source], combined)

    assert not is_combined_cache_stale(combined, [source])


def test_combined_stale_on_source_change(tmp_path):
    """Size and mtime, not "is the output newer", so a rebuilt source of the
    same age must still invalidate the cache."""
    source = str(tmp_path / "source.parquet")
    combined = str(tmp_path / "combined.parquet")
    _write_parquet(source, [_row("conf/x/1")])
    merge_and_dedup_parquets([source], combined)

    _write_parquet(source, [_row("conf/x/1"), _row("conf/x/2")])

    assert is_combined_cache_stale(combined, [source])
