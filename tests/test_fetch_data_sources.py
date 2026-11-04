from __future__ import annotations

import gzip
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

# The script lives in misc/, outside the installed package, so it is loaded by path
_SCRIPT = Path(__file__).resolve().parents[1] / "misc" / "fetch_data_sources.py"
_spec = importlib.util.spec_from_file_location("fetch_data_sources", _SCRIPT)
assert _spec and _spec.loader
fetch_data_sources = importlib.util.module_from_spec(_spec)
sys.modules["fetch_data_sources"] = fetch_data_sources
_spec.loader.exec_module(fetch_data_sources)


DUMP_XML = "<?xml version='1.0'?>\n<dblp></dblp>\n"
DTD_TEXT = "<!ELEMENT dblp ANY>"


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A stand-in dblp.org: URL -> bytes, plus a record of what was requested."""
    archive_bytes = gzip.compress(DUMP_XML.encode())
    served = {
        f"{fetch_data_sources.DBLP_BASE}/dblp.xml.gz": archive_bytes,
        f"{fetch_data_sources.DBLP_BASE}/dblp.dtd": DTD_TEXT.encode(),
        f"{fetch_data_sources.DBLP_BASE}/dblp.xml.gz.md5": (
            hashlib.md5(archive_bytes).hexdigest() + "  dblp.xml.gz"  # noqa: S324
        ).encode(),
    }
    requested: list[str] = []

    def fake_download(url, dest):
        requested.append(url)
        Path(dest).write_bytes(served[url])

    def fake_md5_url(url):
        body = served.get(url)
        return body.decode().split()[0] if body else None

    monkeypatch.setattr(fetch_data_sources, "_download_with_retry", fake_download)
    monkeypatch.setattr(fetch_data_sources, "_published_md5", fake_md5_url)
    return {"served": served, "requested": requested}


def test_fetch_dblp_extracted(remote, tmp_path):
    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)

    assert (tmp_path / "dblp.xml").read_text() == DUMP_XML
    assert (tmp_path / "dblp.dtd").read_text() == DTD_TEXT


def test_second_run_is_noop(remote, tmp_path):
    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)
    remote["requested"].clear()

    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)

    assert remote["requested"] == []


def test_force_refetches_everything(remote, tmp_path):
    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)
    remote["requested"].clear()

    fetch_data_sources.fetch_dblp(tmp_path, force=True, keep_archive=False)

    assert any(url.endswith("dblp.xml.gz") for url in remote["requested"])
    assert any(url.endswith("dblp.dtd") for url in remote["requested"])


def test_missing_dtd_refetched(remote, tmp_path):
    """The common failure. Re-fetching a gigabyte to get a few KB would be absurd."""
    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)
    (tmp_path / "dblp.dtd").unlink()
    remote["requested"].clear()

    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)

    assert (tmp_path / "dblp.dtd").read_text() == DTD_TEXT
    assert not any(url.endswith(".gz") for url in remote["requested"])


def test_checksum_mismatch_raises(remote, tmp_path, monkeypatch):
    """A silently truncated multi-GB download must not reach the extractor."""
    monkeypatch.setattr(fetch_data_sources, "_published_md5", lambda url: "0" * 32)

    with pytest.raises(RuntimeError, match="failed its checksum"):
        fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)

    assert not (tmp_path / "dblp.xml.gz").exists()
    assert not (tmp_path / "dblp.xml").exists()


def test_unreachable_checksum_ok(remote, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data_sources, "_published_md5", lambda url: None)

    fetch_data_sources.fetch_dblp(tmp_path, force=False, keep_archive=False)

    assert (tmp_path / "dblp.xml").read_text() == DUMP_XML


def test_extract_failure_leaves_no_partial(tmp_path):
    archive = tmp_path / "dblp.xml.gz"
    archive.write_bytes(b"not gzip at all")

    with pytest.raises(OSError):
        fetch_data_sources._extract(archive, tmp_path / "dblp.xml")

    assert not (tmp_path / "dblp.xml").exists()
    assert not (tmp_path / "dblp.xml.partial").exists()


def test_no_key_dblp_only():
    """No key means "just DBLP", not an error: S2 is an optional extra source."""
    assert fetch_data_sources.chosen_datasets(None, None) == ["dblp"]


def test_key_enables_s2():
    assert fetch_data_sources.chosen_datasets(None, "k") == [
        "dblp",
        "authors",
        "papers",
    ]
