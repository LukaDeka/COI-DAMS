from __future__ import annotations

import io
import json
import urllib.parse

import pytest

from coi_dams import webhdfs
from coi_dams.webhdfs import _endpoint, delete_dir, download_dir, uri_to_path


class _FakeResponse(io.BytesIO):
    """Just enough of an http.client.HTTPResponse for these call sites."""

    def __init__(self, body: bytes = b"", status: int = 200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class _Recorder:
    """Records every request the module makes and replays canned replies."""

    DATANODE_URL = "http://dn:9864/write?token=x"

    def __init__(self):
        self.made: list[dict] = []
        self.replies: list[_FakeResponse] = []
        self.redirect = _FakeResponse(
            status=307, headers={"Location": self.DATANODE_URL}
        )

    def __call__(self, url, *, method="GET", data=None, headers=None):
        self.made.append({"url": url, "method": method, "data": data})
        return self.replies.pop(0) if self.replies else _FakeResponse()

    def open(self, request, timeout=None):
        """Stands in for the opener built to keep a 307 unfollowed."""
        self.made.append(
            {"url": request.full_url, "method": request.get_method(), "data": None}
        )
        return self.redirect

    @property
    def ops(self) -> list[str]:
        """The `op` of each request, in order; "" for a bare DataNode URL."""
        return [
            urllib.parse.parse_qs(urllib.parse.urlsplit(call["url"]).query).get(
                "op", [""]
            )[0]
            for call in self.made
        ]


@pytest.fixture
def calls(monkeypatch):
    """Both outbound paths of the module, stubbed together.

    build_opener is stubbed here rather than per-test so that no transfer
    test can reach the network by omission - forgetting it would otherwise
    send a real PUT to whatever host the test names.
    """
    recorder = _Recorder()
    monkeypatch.setattr(webhdfs, "_open", recorder)
    monkeypatch.setattr(webhdfs.urllib.request, "build_opener", lambda *_args: recorder)
    return recorder


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("hdfs://10.10.10.10:9000/coi/dblp.parquet", "/coi/dblp.parquet"),
        ("hdfs://host/coi", "/coi"),
        ("hdfs://host:9000/a/b/c.csv", "/a/b/c.csv"),
    ],
)
def test_uri_to_path(uri, expected):
    assert uri_to_path(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "hdfs://10.10.10.10:9000",  # Host only
        "hdfs://",
        "/coi/dblp.parquet",  # No scheme
        "",
    ],
)
def test_uri_to_path_bad_uri(uri):
    with pytest.raises(ValueError) as error:
        uri_to_path(uri)
    assert "--hdfs-parquet" in str(error.value)


def test_endpoint_escaping():
    url = _endpoint(
        "http://nn:9870/", "/coi/my dump.parquet", "CREATE", overwrite="true"
    )
    parts = urllib.parse.urlsplit(url)
    assert parts.path == "/webhdfs/v1/coi/my%20dump.parquet"
    query = urllib.parse.parse_qs(parts.query)
    assert query["op"] == ["CREATE"]
    assert query["overwrite"] == ["true"]
    assert query["user.name"]


def _liststatus(*entries) -> _FakeResponse:
    """A LISTSTATUS reply listing (name, type) pairs."""
    return _FakeResponse(
        json.dumps(
            {
                "FileStatuses": {
                    "FileStatus": [
                        {"pathSuffix": name, "type": kind} for name, kind in entries
                    ]
                }
            }
        ).encode()
    )


def test_upload_follows_redirect(calls, tmp_path):
    source = tmp_path / "cache.parquet"
    source.write_bytes(b"payload")
    calls.replies = [_FakeResponse(), _FakeResponse(status=201)]

    webhdfs.upload_file(str(source), "/coi/cache.parquet", "http://nn:9870")

    # MKDIRS on the parent, the CREATE that answers 307, then the body PUT at the DataNode
    assert calls.ops[:2] == ["MKDIRS", "CREATE"]
    assert "/webhdfs/v1/coi?" in calls.made[0]["url"]
    assert calls.made[2]["url"] == calls.DATANODE_URL
    assert calls.made[2]["method"] == "PUT"


def test_upload_no_redirect(calls, tmp_path):
    """A 200 here means the NameNode wrote nothing and the body went nowhere."""
    source = tmp_path / "cache.parquet"
    source.write_bytes(b"payload")
    calls.redirect = _FakeResponse(status=200)

    with pytest.raises(RuntimeError, match="didn't redirect"):
        webhdfs.upload_file(str(source), "/coi/cache.parquet", "http://nn:9870")


def test_upload_put_fails(calls, tmp_path):
    source = tmp_path / "cache.parquet"
    source.write_bytes(b"payload")
    calls.replies = [_FakeResponse(), _FakeResponse(status=403)]

    with pytest.raises(RuntimeError, match="status 403"):
        webhdfs.upload_file(str(source), "/coi/cache.parquet", "http://nn:9870")


def test_download_dir_part_files(calls, tmp_path):
    calls.replies = [
        _liststatus(
            ("part-00000.csv", "FILE"),
            ("part-00001.csv", "FILE"),
            ("_SUCCESS", "FILE"),  # Spark's completion marker, not data
            ("nested", "DIRECTORY"),
        ),
        _FakeResponse(b"a,1\n"),
        _FakeResponse(b"b,2\n"),
    ]
    local = tmp_path / "pulled"

    download_dir("hdfs://nn:9000/coi/csv_out/coi", str(local), "http://nn:9870")

    assert calls.ops == ["LISTSTATUS", "OPEN", "OPEN"]
    assert sorted(p.name for p in local.iterdir()) == [
        "part-00000.csv",
        "part-00001.csv",
    ]
    assert (local / "part-00000.csv").read_text() == "a,1\n"


def test_delete_dir_recursive(calls):
    delete_dir("hdfs://nn:9000/coi/csv_out/coi", "http://nn:9870")

    assert calls.ops == ["DELETE"]
    assert calls.made[0]["method"] == "DELETE"
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls.made[0]["url"]).query)
    assert query["recursive"] == ["true"]
