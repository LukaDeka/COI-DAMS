from __future__ import annotations

import csv

import pytest

from coi_dams import outputs
from coi_dams.constants import DISAM_CERTAIN, DISAM_DOUBT, DISAM_FAILED
from coi_dams.outputs import (
    write_analysis_csv,
    write_coi_csv,
    write_unresolved_users_csv,
)
from coi_dams.users import Disambiguation, ResolvedUser, User
from conftest import requires_java

pytestmark = requires_java

# One row per scoring band, so the threshold filter has both sides
SCORED_ROWS = [
    {
        "email_a": "a@example.org",
        "email_b": "b@example.org",
        "author_a": "Ann Alpha 0001",
        "author_b": "Bob Beta",
        "dblp_id_a": "a/Alpha",
        "dblp_id_b": "b/Beta",
        "disam_level_a": DISAM_CERTAIN,
        "disam_level_b": DISAM_CERTAIN,
        "coi_score": 200,
        "depth": 1,
        "coi_type": "co-author",
        "evidence_path_count": 2,
        "evidence_papers": "p1; p3",
    },
    {
        "email_a": "c@example.org",
        "email_b": "d@example.org",
        "author_a": "Cris Gamma",
        "author_b": "Dee Van Delta",
        "dblp_id_a": "c/Gamma",
        "dblp_id_b": "d/Delta",
        "disam_level_a": DISAM_CERTAIN,
        "disam_level_b": DISAM_FAILED,
        "coi_score": 2,
        "depth": 3,
        "coi_type": "submarine",
        "evidence_path_count": 1,
        "evidence_papers": "p9",
    },
]


@pytest.fixture
def scored(spark):
    return spark.createDataFrame(SCORED_ROWS)


def _rows(path) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def test_coi_csv_threshold_filter(scored, tmp_path):
    out = tmp_path / "coi.csv"

    write_coi_csv(scored, str(out), threshold=200)

    # Tab-separated and headerless, the 2-point pair is dropped
    assert out.read_text(encoding="utf-8") == "a@example.org\tb@example.org\n"


def test_coi_csv_threshold_inclusive(scored, tmp_path):
    """A depth-3 pair scoring exactly the threshold must survive."""
    out = tmp_path / "coi.csv"

    write_coi_csv(scored, str(out), threshold=2)

    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_analysis_csv_header_and_names(scored, tmp_path):
    out = tmp_path / "analysis.csv"

    write_analysis_csv(scored, str(out))

    rows = _rows(out)
    header = rows[0]
    assert header[0] == "email_a"
    assert "disambiguation_total" in header

    by_email = {
        row[header.index("email_a")]: dict(zip(header, row, strict=True))
        for row in rows[1:]
    }
    top = by_email["a@example.org"]
    # The DBLP homonym suffix is presentation noise, dblp_id carries identity
    assert top["full_name_a"] == "Ann Alpha"
    assert top["name_a"] == "Ann"
    assert top["surname_a"] == "Alpha"
    # A multi-word given name keeps everything but the last token
    assert by_email["c@example.org"]["name_b"] == "Dee Van"
    assert by_email["c@example.org"]["surname_b"] == "Delta"


def test_analysis_csv_weaker_confidence(scored, tmp_path):
    out = tmp_path / "analysis.csv"

    write_analysis_csv(scored, str(out))

    rows = _rows(out)
    header = rows[0]
    confidence = {
        row[header.index("email_a")]: row[header.index("disambiguation_total")]
        for row in rows[1:]
    }
    assert confidence["a@example.org"] == DISAM_CERTAIN
    assert confidence["c@example.org"] == DISAM_FAILED  # Certain + failed


def test_valid_csv_escape(spark, tmp_path):
    author = 'John "Nickname", Doe; Jr/154/6778'
    df = spark.createDataFrame([("a", author)], ["id", "author"])
    out = tmp_path / "q.csv"
    outputs._write_csv(df, out, None, header="id,author")
    assert _rows(out) == [["id", "author"], ["a", author]]


def test_write_csv_hdfs(scored, tmp_path, monkeypatch):
    """Cluster mode: Spark writes to HDFS, the driver downloads and cleans up."""
    downloaded: list[tuple[str, str]] = []
    deleted: list[str] = []

    def fake_download(hdfs_dir_uri, local_dir, webhdfs_base):
        downloaded.append((hdfs_dir_uri, local_dir))
        # Stand in for the real transfer, Spark already wrote the parts there

    monkeypatch.setattr(outputs.webhdfs, "download_dir", fake_download)
    monkeypatch.setattr(
        outputs.webhdfs, "delete_dir", lambda uri, base: deleted.append(uri)
    )

    staging = tmp_path / "hdfs_staging"
    out = tmp_path / "coi.csv"
    write_coi_csv(
        scored, str(out), threshold=0, hdfs_tmp=(str(staging), "http://nn:9870")
    )

    assert downloaded and downloaded[0][0] == str(staging)
    assert deleted == [str(staging)]


def test_write_csv_download_fails(scored, tmp_path, monkeypatch):
    monkeypatch.setattr(
        outputs.webhdfs,
        "download_dir",
        lambda *_: (_ for _ in ()).throw(OSError("truncated")),
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        outputs.webhdfs, "delete_dir", lambda uri, base: deleted.append(uri)
    )

    staging = tmp_path / "hdfs_staging"
    with pytest.raises(OSError, match="truncated"):
        write_coi_csv(
            scored,
            str(tmp_path / "coi.csv"),
            threshold=0,
            hdfs_tmp=(str(staging), "http://nn:9870"),
        )

    assert deleted == [str(staging)]


def test_write_csv_cleanup_fails(scored, tmp_path, monkeypatch):
    """A failed cleanup is logged, not raised; the output is already written."""
    monkeypatch.setattr(outputs.webhdfs, "download_dir", lambda *_: None)
    monkeypatch.setattr(
        outputs.webhdfs,
        "delete_dir",
        lambda *_: (_ for _ in ()).throw(OSError("permission denied")),
    )

    out = tmp_path / "coi.csv"
    write_coi_csv(
        scored,
        str(out),
        threshold=0,
        hdfs_tmp=(str(tmp_path / "staging"), "http://nn:9870"),
    )

    assert out.exists()


def _resolved(email: str, level: str) -> ResolvedUser:
    user = User(
        first_name="Ann",
        last_name="Alpha",
        email=email,
        organization="Example Uni",
        country="NL",
        dblp_url="",
        dblp_pid=None,
        gscholar_url="",
        semscholar_url="",
        openreview_id="",
    )
    return ResolvedUser(user=user, disambiguation=Disambiguation(set(), level, ""))


@pytest.mark.parametrize("level", [DISAM_FAILED, DISAM_DOUBT])
def test_unresolved_csv_rows(level, tmp_path):
    out = tmp_path / "nested" / "unresolved.csv"

    write_unresolved_users_csv([_resolved("a@example.org", level)], str(out))

    rows = _rows(out)
    assert rows[0] == [
        "email",
        "full_name",
        "organization",
        "country",
        "disambiguation",
        "reason",
    ]
    assert rows[1][0] == "a@example.org"
    assert rows[1][4] == level
    # The reason column makes the file actionable by hand
    assert "check manually" in rows[1][5]
