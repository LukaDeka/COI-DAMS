"""Reviewer qualification: has a user published enough at major venues?

A reviewer qualifies on REVIEWER_MIN_PAPERS papers across QUALIFYING_VENUES.
The algorithm is simple, has linear runtime, so it doesn't need to run on a cluster.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq

from .constants import (
    DISAM_CERTAIN,
    DISAM_DOUBT,
    DISAM_FAILED,
    DOTENV_PATH,
    REVIEWER_MIN_PAPERS,
)
from .dblp import homepages_path, load_homepages
from .outputs import write_unresolved_users_csv
from .users import (
    Disambiguation,
    ResolvedUser,
    User,
    build_dblp_indexes,
    disambiguate,
    load_users_csv,
    unresolved_users,
)

logger = logging.getLogger(__name__)

_KEYWORD_SEPARATOR = ","


# Parses one `LABEL: keyword, keyword` entry per line into a label -> keywords map
def _env_venues(name: str, default: dict[str, list[str]]) -> dict[str, list[str]]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    venues: dict[str, list[str]] = {}
    for entry in raw.splitlines():
        entry = entry.strip()
        if not entry:
            continue
        label, sep, keywords = entry.partition(":")
        if not sep or not label.strip() or not keywords.strip():
            raise SystemExit(
                f"{name} entry {entry!r} must look like "
                f"'LABEL:keyword{_KEYWORD_SEPARATOR}keyword', "
                f"check the environment or {DOTENV_PATH}"
            )
        venues[label.strip()] = [
            kw.strip() for kw in keywords.split(_KEYWORD_SEPARATOR) if kw.strip()
        ]
    return venues


# Venue label -> substrings that identify it in a DBLP journal/booktitle string.
# Configurable via COI_QUALIFYING_VENUES (see .env.example)
QUALIFYING_VENUES: dict[str, list[str]] = _env_venues(
    "COI_QUALIFYING_VENUES",
    {
        "SIGMOD": ["sigmod", "management of data"],
        "VLDB": ["vldb", "very large data bases", "pvldb", "proc. vldb"],
        "ICDE": ["icde", "data engineering"],
        "EDBT": ["edbt", "extending database technology"],
        "CIDR": ["cidr", "innovative data systems research"],
    },
)

PARQUET_BATCH_SIZE = 50_000

_BASE_COLUMNS = ["first_name", "last_name", "email", "dblp_url"]
_SUMMARY_COLUMNS = [
    "total_qualifying_papers",
    "total_publication_count",
    "disambiguation_level",
]

# What makes two DBLP records the same paper (title, doi).
PaperIdentity = tuple[str, str]


class Paper(NamedTuple):
    title: str
    doi: str
    venue_label: str


class PublicationIndex(NamedTuple):
    """Per-author publication facts, gathered in one pass over the Parquet."""

    qualifying_papers: dict[
        str, list[Paper]
    ]  # Author name -> papers at qualifying venues
    paper_identities: dict[
        str, set[PaperIdentity]
    ]  # Author name -> distinct papers, any venue


def _paper_identity(title: str, doi: str) -> PaperIdentity:
    return (title.lower(), doi)


# The QUALIFYING_VENUES label a DBLP venue string belongs to, if any
def _venue_label(venue: str) -> str | None:
    lowered = venue.lower()
    for label, keywords in QUALIFYING_VENUES.items():
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def build_publication_index(
    parquet_path: str, aliases_of_interest: set[str]
) -> PublicationIndex:
    """Index the publications of `aliases_of_interest`, ignoring every other author.

    Scoped to the aliases the report asks about, a few thousand names rather
    than the ~3M in DBLP.
    """
    qualifying_papers: dict[str, list[Paper]] = {}
    paper_identities: dict[str, set[PaperIdentity]] = {}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_SIZE, columns=["title", "authors", "venue", "doi"]
    ):
        columns = batch.to_pydict()
        for title, authors, venue, doi in zip(
            columns["title"],
            columns["authors"],
            columns["venue"],
            columns["doi"],
            strict=True,
        ):
            relevant = [author for author in authors if author in aliases_of_interest]
            if not relevant:
                continue
            label = _venue_label(venue)
            identity = _paper_identity(title, doi)
            for author in relevant:
                paper_identities.setdefault(author, set()).add(identity)
                if label is not None:
                    qualifying_papers.setdefault(author, []).append(
                        Paper(title, doi, label)
                    )
        del columns
    gc.collect()
    return PublicationIndex(qualifying_papers, paper_identities)


# Qualifying papers per venue, deduplicated by title+DOI across aliases.
def _papers_per_venue(
    aliases: set[str], index: PublicationIndex
) -> dict[str, list[Paper]]:
    per_venue: dict[str, list[Paper]] = {label: [] for label in QUALIFYING_VENUES}
    seen: set[PaperIdentity] = set()
    for alias in aliases:
        for paper in index.qualifying_papers.get(alias, []):
            identity = _paper_identity(paper.title, paper.doi)
            if identity in seen:
                continue
            seen.add(identity)
            per_venue[paper.venue_label].append(paper)
    return per_venue


# Distinct papers over all aliases, deduplicated the same way as _papers_per_venue.
def _total_publications(aliases: set[str], index: PublicationIndex) -> int:
    return len(
        {
            identity
            for alias in aliases
            for identity in index.paper_identities.get(alias, ())
        }
    )


def _format_papers(papers: list[Paper]) -> str:
    return "; ".join(f"{p.title} [{p.doi}]" if p.doi else p.title for p in papers)


def _report_row(
    user: User,
    match: Disambiguation,
    per_venue: dict[str, list[Paper]],
    index: PublicationIndex,
) -> dict[str, str | int]:
    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "dblp_url": user.dblp_url,
        **{label: _format_papers(papers) for label, papers in per_venue.items()},
        "total_qualifying_papers": sum(len(papers) for papers in per_venue.values()),
        "total_publication_count": _total_publications(match.aliases, index),
        "disambiguation_level": match.level,
    }


def _write_report(rows: list[dict[str, str | int]], output_csv_path: str) -> None:
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=[*_BASE_COLUMNS, *QUALIFYING_VENUES, *_SUMMARY_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(rows)


def check_reviewer_qualifications(
    parquet_path: str,
    users_csv_path: str,
    output_csv_path: str,
    reverse: bool = False,
    homepages_parquet: str | None = None,
    unresolved_output_path: str = "unresolved-users.csv",
) -> None:
    # TODO: refactor this so BOTH files are written. also remove --reverse flag entirely. output failed disambiguation peers in a different file. update flags
    """
    Write one CSV row per qualifying reviewer, or per failing one if `reverse`.

    Columns: first_name, last_name, email, dblp_url, one per qualifying venue,
    total_qualifying_papers, total_publication_count, disambiguation_level.

    Users are disambiguated before the Parquet is scanned, so the scan only has
    to index the aliases those matches produced.

    Users whose disambiguation fails are left out of the scan, since merging
    every same-name DBLP entity attributes other people's publications to them.
    They are written to `unresolved_output_path`, like `coi` does.

    `homepages_parquet` defaults to the index beside `parquet_path`, a caller
    passes the S2-enriched one when it is in use.
    """
    homepages = load_homepages(homepages_parquet or homepages_path(parquet_path))
    logger.info("%s homepage records loaded", f"{len(homepages):,}")

    indexes = build_dblp_indexes(homepages)
    logger.info(
        "%s URLs indexed, %s base names indexed",
        f"{len(indexes.by_url):,}",
        f"{len(indexes.by_name):,}",
    )

    users = load_users_csv(users_csv_path)
    logger.info("Loaded %s reviewer-role users from %s", len(users), users_csv_path)

    resolved = [
        ResolvedUser(user, disambiguate(user, homepages, indexes)) for user in users
    ]
    write_unresolved_users_csv(unresolved_users(resolved), unresolved_output_path)

    matches = [
        (entry.user, entry.disambiguation)
        for entry in resolved
        if entry.disambiguation.level != DISAM_FAILED
    ]
    aliases_of_interest = {alias for _, match in matches for alias in match.aliases}

    logger.info(
        "Scanning Parquet for the publications of %s DBLP alias(es) ...",
        f"{len(aliases_of_interest):,}",
    )
    publications = build_publication_index(parquet_path, aliases_of_interest)

    rows: list[dict[str, str | int]] = []
    level_counts = {DISAM_CERTAIN: 0, DISAM_DOUBT: 0, DISAM_FAILED: 0}
    for entry in resolved:
        level_counts[entry.disambiguation.level] += 1

    for user, match in matches:
        per_venue = _papers_per_venue(match.aliases, publications)
        is_qualified = (
            sum(len(papers) for papers in per_venue.values()) >= REVIEWER_MIN_PAPERS
        )
        if is_qualified != reverse:
            rows.append(_report_row(user, match, per_venue, publications))

    logger.info(
        "%s %s reviewer(s) found", len(rows), "unqualified" if reverse else "qualified"
    )
    logger.info(
        "Disambiguation: certain=%s, doubt=%s, failed=%s (failed users excluded from "
        "the scan, listed in %s)",
        level_counts[DISAM_CERTAIN],
        level_counts[DISAM_DOUBT],
        level_counts[DISAM_FAILED],
        unresolved_output_path,
    )

    _write_report(rows, output_csv_path)
    logger.info("Written -> %s", output_csv_path)
