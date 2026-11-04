"""Driver-side HDFS transfers over the WebHDFS REST API.

Used instead of pyarrow's HadoopFileSystem, which needs a libjvm/Hadoop
classpath the driver machine does not have. Files cross between the driver's
local disk and HDFS: the Parquet cache, the mapping tables and Spark's CSV output.

Authentication is WebHDFS's pseudo mode: the local username is sent as
`user.name` over plain HTTP. A Kerberised deployment would need SPNEGO instead.
"""

from __future__ import annotations

import getpass
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

# Without a timeout an unreachable NameNode hangs the run with no output
REQUEST_TIMEOUT_SECONDS = 60

# Bodies are streamed in blocks because the Parquet cache is multi-GB
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


class _KeepRedirectResponse(urllib.request.HTTPErrorProcessor):
    """Hands 3xx responses back to the caller instead of following them.

    A WebHDFS CREATE answers 307 with the DataNode to write to; the default
    opener re-issues the PUT there but drops the request body, silently
    uploading an empty file. Following it manually keeps the body.
    """

    def http_response(self, request, response):
        return response

    https_response = http_response


def uri_to_path(hdfs_uri: str) -> str:
    """Strip the "hdfs://host:port" prefix off a URI, leaving the bare path.

    WebHDFS addresses paths, while Spark and the CLI flags use full URIs.
    """
    parts = urllib.parse.urlsplit(hdfs_uri)
    if not parts.scheme or not parts.netloc or not parts.path:
        raise ValueError(
            f"expected an hdfs://host:port/path URI, got {hdfs_uri!r} "
            "(check --hdfs-parquet, --hdfs and --checkpoint-dir)"
        )
    return parts.path


def _endpoint(webhdfs_base: str, path: str, operation: str, **params: str) -> str:
    # The path can carry spaces or non-ASCII, so both halves are escaped
    query = urllib.parse.urlencode(
        {"op": operation, "user.name": getpass.getuser(), **params}
    )
    quoted_path = urllib.parse.quote(path)
    return f"{webhdfs_base.rstrip('/')}/webhdfs/v1{quoted_path}?{query}"


def _open(
    url: str, *, method: str = "GET", data=None, headers: dict[str, str] | None = None
):
    request = urllib.request.Request(url, data=data, method=method)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)


def upload_file(local_path: str, hdfs_path: str, webhdfs_base: str) -> None:
    """Copy one local file to `hdfs_path`, creating parent directories."""
    # PurePosixPath because HDFS paths are POSIX whatever the driver runs on
    mkdirs_url = _endpoint(webhdfs_base, str(PurePosixPath(hdfs_path).parent), "MKDIRS")
    with _open(mkdirs_url, method="PUT") as response:
        response.read()

    create_url = _endpoint(webhdfs_base, hdfs_path, "CREATE", overwrite="true")
    opener = urllib.request.build_opener(_KeepRedirectResponse)
    with opener.open(
        urllib.request.Request(create_url, method="PUT"),
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as redirect:
        if redirect.status != 307:
            raise RuntimeError(
                f"WebHDFS CREATE for {hdfs_path} didn't redirect to a DataNode "
                f"(status {redirect.status})"
            )
        datanode_url = redirect.headers["Location"]

    source = Path(local_path)
    # Content-Length is set by hand, urllib would send chunked and WebHDFS rejects that.
    # Nested `with` because the parenthesized form is 3.10 syntax
    with source.open("rb") as body:  # noqa: SIM117
        with _open(
            datanode_url,
            method="PUT",
            data=body,
            headers={"Content-Length": str(source.stat().st_size)},
        ) as response:
            if response.status not in (200, 201):
                raise RuntimeError(
                    f"WebHDFS PUT for {hdfs_path} failed (status {response.status})"
                )


def download_dir(hdfs_dir_uri: str, local_dir: str, webhdfs_base: str) -> None:
    """Copy every file directly under an HDFS directory into `local_dir`.

    Fetches Spark's multi-part output back to the driver: on a real cluster the
    executors write those parts to HDFS, which is the only filesystem both
    sides can see.
    """
    hdfs_dir = uri_to_path(hdfs_dir_uri)
    list_url = _endpoint(webhdfs_base, hdfs_dir, "LISTSTATUS")
    with _open(list_url) as response:
        statuses = json.loads(response.read())["FileStatuses"]["FileStatus"]

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    for status in statuses:
        name = status["pathSuffix"]
        # Only the part files carry data, not _SUCCESS and friends
        if status["type"] != "FILE" or name.startswith("_"):
            continue
        open_url = _endpoint(webhdfs_base, f"{hdfs_dir}/{name}", "OPEN")
        with _open(open_url) as response, (local / name).open("wb") as out_file:
            shutil.copyfileobj(response, out_file, _COPY_CHUNK_BYTES)


def delete_dir(hdfs_dir_uri: str, webhdfs_base: str) -> None:
    """Remove an HDFS directory and its contents.

    Cleans up the staging directory Spark writes its CSV parts to; without this
    every run leaves another copy of the full output behind on the cluster.
    """
    delete_url = _endpoint(
        webhdfs_base, uri_to_path(hdfs_dir_uri), "DELETE", recursive="true"
    )
    with _open(delete_url, method="DELETE") as response:
        response.read()
