from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coi_dams import cache

SCHEMA = pa.schema([pa.field("a", pa.string())])


def _write(path, manifest: cache.Manifest | None = None, rows=("1",)) -> str:
    table = pa.Table.from_pylist([{"a": value} for value in rows], schema=SCHEMA)
    writer = pq.ParquetWriter(
        str(path), cache.schema_with_manifest(SCHEMA, manifest), compression="snappy"
    )
    writer.write_table(table)
    writer.close()
    return str(path)


def test_fingerprint_order(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("one")
    second.write_text("two")

    assert cache.fingerprint([str(first), str(second)]) == cache.fingerprint(
        [str(second), str(first)]
    )


def test_fingerprint_directory(tmp_path):
    source = tmp_path / "parts"
    source.mkdir()
    (source / "part-00.jsonl.gz").write_bytes(b"a")
    (source / "part-01.jsonl.gz").write_bytes(b"bb")

    prints = cache.fingerprint([str(source)])

    assert [p["size"] for p in prints] == [1, 2]


def test_a_missing_artefact_is_stale(tmp_path):
    assert cache.is_stale(str(tmp_path / "absent.parquet"), [])


def test_no_manifest_is_stale(tmp_path):
    path = _write(tmp_path / "x.parquet")
    assert cache.is_stale(path, [])


def test_matching_sources_fresh(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("one")
    path = _write(tmp_path / "x.parquet", cache.build_manifest([str(source)]))

    assert not cache.is_stale(path, [str(source)])


def test_changed_source_is_stale(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("one")
    path = _write(tmp_path / "x.parquet", cache.build_manifest([str(source)]))

    source.write_text("a different length entirely")

    assert cache.is_stale(path, [str(source)])


def test_manifest_round_trip(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("one")
    manifest = cache.build_manifest([str(source)], {cache.PARAM_CUTOFF_YEAR: 2016})
    path = _write(tmp_path / "x.parquet", manifest)

    loaded = cache.read_manifest(path)

    assert loaded is not None
    assert loaded.params == {cache.PARAM_CUTOFF_YEAR: 2016}
    assert loaded.sources == manifest.sources
    assert loaded.built_at == manifest.built_at


def test_manifest_version_mismatch(tmp_path, monkeypatch):
    path = _write(tmp_path / "x.parquet", cache.build_manifest([]))
    monkeypatch.setattr(cache, "FORMAT_VERSION", cache.FORMAT_VERSION + 1)

    assert cache.read_manifest(path) is None


def test_matching_params_ok(tmp_path):
    params = {cache.PARAM_CUTOFF_YEAR: 2016}
    path = _write(tmp_path / "x.parquet", cache.build_manifest([], params))

    cache.check_parameters(path, params, force=False, description="x")  # No raise


def test_param_mismatch_raises(tmp_path):
    path = _write(
        tmp_path / "x.parquet",
        cache.build_manifest([], {cache.PARAM_CUTOFF_YEAR: 2016}),
    )

    with pytest.raises(SystemExit) as error:
        cache.check_parameters(
            path,
            {cache.PARAM_CUTOFF_YEAR: 2010},
            force=False,
            description="The publications cache",
        )

    message = str(error.value)
    assert "The publications cache" in message
    assert cache.PARAM_CUTOFF_YEAR in message
    assert "2016" in message and "2010" in message
    assert "--force" in message


def test_force_skips_check(tmp_path):
    """The artefact is about to be rebuilt, so its old contents don't matter."""
    path = _write(
        tmp_path / "x.parquet",
        cache.build_manifest([], {cache.PARAM_CUTOFF_YEAR: 2016}),
    )

    cache.check_parameters(
        path, {cache.PARAM_CUTOFF_YEAR: 2010}, force=True, description="x"
    )


def test_unverifiable_cache_refused(tmp_path):
    """No manifest means the cutoff it was built under is unknown, not 'fine'."""
    path = _write(tmp_path / "x.parquet")

    with pytest.raises(SystemExit, match="no readable cache manifest"):
        cache.check_parameters(
            path, {cache.PARAM_CUTOFF_YEAR: 2016}, force=False, description="x"
        )
