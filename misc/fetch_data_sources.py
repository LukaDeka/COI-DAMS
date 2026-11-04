#!/usr/bin/env python3
"""
Usage:
    # DBLP only
    ./misc/fetch_data_sources.py

    # DBLP + Semantic Scholar
    export S2_API_KEY=...   # https://www.semanticscholar.org/product/api
    ./misc/fetch_data_sources.py

    # Only one dataset at a specified directory
    ./misc/fetch_data_sources.py --datasets papers --out-dir /mnt/big

Re-running skips whatever is already complete.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("fetch_data_sources")

API_BASE = "https://api.semanticscholar.org/datasets/v1"
_CHUNK_SIZE = 1 << 20  # 1 MiB
_MAX_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 5

DBLP_BASE = "https://dblp.org/xml"
DBLP_ARCHIVE = "dblp.xml.gz"
DBLP_DUMP = "dblp.xml"
DBLP_DTD = "dblp.dtd"

S2_DATASETS = ("authors", "papers")
ALL_DATASETS = ("dblp", *S2_DATASETS)


# Streams to a .partial file and renames, so an interrupted download doesn't break things
def _download_with_retry(url: str, dest: Path) -> None:
    partial = dest.with_suffix(dest.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                open(partial, "wb") as fh,
            ):
                while chunk := response.read(_CHUNK_SIZE):
                    fh.write(chunk)
            partial.rename(dest)
            return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            logger.warning(
                "download attempt %d/%d for %s failed: %s",
                attempt,
                _MAX_ATTEMPTS,
                dest.name,
                exc,
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    partial.unlink(missing_ok=True)
    raise RuntimeError(
        f"giving up on {dest.name} after {_MAX_ATTEMPTS} attempts"
    ) from last_error


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


# The archive's checksum from dblp.org, or None if fetch fails
def _published_md5(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode().split()[0]
    except (urllib.error.URLError, OSError, IndexError, UnicodeDecodeError) as exc:
        logger.warning("could not fetch %s (%s); skipping checksum check", url, exc)
        return None


# Gunzips `archive` to `dest`
def _extract(archive: Path, dest: Path) -> None:
    partial = dest.with_suffix(dest.suffix + ".partial")
    logger.info("extracting %s -> %s (~4GB) ...", archive.name, dest.name)
    try:
        with gzip.open(archive, "rb") as source, open(partial, "wb") as target:
            shutil.copyfileobj(source, target, _CHUNK_SIZE)
        partial.rename(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def fetch_dblp(out_dir: Path, force: bool, keep_archive: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / DBLP_ARCHIVE
    dump = out_dir / DBLP_DUMP
    dtd = out_dir / DBLP_DTD

    if not dump.exists() or force:
        if not archive.exists() or force:
            logger.info("downloading %s (~1GB) ...", archive.name)
            _download_with_retry(f"{DBLP_BASE}/{DBLP_ARCHIVE}", archive)
        else:
            logger.info("%s already downloaded, reusing it", archive.name)

        expected = _published_md5(f"{DBLP_BASE}/{DBLP_ARCHIVE}.md5")
        if expected:
            actual = _md5(archive)
            if actual != expected:
                archive.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{DBLP_ARCHIVE} failed its checksum (expected {expected}, "
                    f"got {actual}). The download was removed, re-run to retry"
                )
            logger.info("%s checksum verified", archive.name)

        _extract(archive, dump)
        if not keep_archive:
            archive.unlink(missing_ok=True)
            logger.info("removed %s (pass --keep-archive to keep it)", archive.name)
    else:
        logger.info("%s already present, skipping", dump.name)

    # Fetched independently of the dump
    if not dtd.exists() or force:
        logger.info("downloading %s ...", dtd.name)
        _download_with_retry(f"{DBLP_BASE}/{DBLP_DTD}", dtd)
    else:
        logger.info("%s already present, skipping", dtd.name)

    logger.info("DBLP ready: pass --dblp-xml %s", dump)


# GET api.semanticscholar.org/datasets/v1 endpoint with the API key header
def _api_get(path: str, api_key: str) -> dict:
    request = urllib.request.Request(
        f"{API_BASE}{path}", headers={"x-api-key": api_key}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve_release(release: str, api_key: str) -> str:
    """Resolve "latest" to a YYYY-MM-DD release ID"""
    if release != "latest":
        return release
    release_ids = _api_get("/release/latest", api_key)
    if isinstance(release_ids, list):
        return release_ids[-1]
    return release_ids["release_id"]


def dataset_file_urls(release_id: str, dataset_name: str, api_key: str) -> list[str]:
    info = _api_get(f"/release/{release_id}/dataset/{dataset_name}", api_key)
    return info["files"]


def fetch_s2_dataset(
    release_id: str, dataset_name: str, out_dir: Path, api_key: str, force: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = dataset_file_urls(release_id, dataset_name, api_key)
    logger.info("%s/%s: %d file(s)", release_id, dataset_name, len(urls))
    for index, url in enumerate(urls):
        dest = out_dir / f"part-{index:05d}.jsonl.gz"
        if dest.exists() and not force:
            logger.info(
                "[%d/%d] %s already present, skipping", index + 1, len(urls), dest.name
            )
            continue
        logger.info("[%d/%d] downloading %s", index + 1, len(urls), dest.name)
        _download_with_retry(url, dest)


def chosen_datasets(requested: list[str] | None, api_key: str | None) -> list[str]:
    if requested:
        return requested
    return ["dblp", *S2_DATASETS] if api_key else ["dblp"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Base directory for every dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=None,
        help="Which dataset(s) to fetch (default: dblp, plus authors and "
        "papers when an API key is available)",
    )
    parser.add_argument(
        "--release",
        default="latest",
        help="Semantic Scholar release id (YYYY-MM-DD) or 'latest' (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("S2_API_KEY"),
        help="Semantic Scholar API key (default: $S2_API_KEY); not needed for dblp",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep dblp.xml.gz after extracting it (default: remove it, since "
        "the extracted dump is what the pipeline reads)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract files that already exist locally",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    datasets = chosen_datasets(args.datasets, args.api_key)
    s2_wanted = [name for name in datasets if name in S2_DATASETS]

    if s2_wanted and not args.api_key:
        parser.error(
            f"no API key: {', '.join(s2_wanted)} need --api-key or $S2_API_KEY "
        )
    if not s2_wanted:
        logger.info("no Semantic Scholar API key: fetching DBLP only")

    out_dir = Path(args.out_dir)
    if "dblp" in datasets:
        fetch_dblp(out_dir, args.force, args.keep_archive)

    if s2_wanted:
        release_id = resolve_release(args.release, args.api_key)
        logger.info("using Semantic Scholar release %s", release_id)
        for dataset_name in s2_wanted:
            fetch_s2_dataset(
                release_id,
                dataset_name,
                out_dir / "semantic_scholar" / dataset_name,
                args.api_key,
                args.force,
            )

    logger.info("done -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
