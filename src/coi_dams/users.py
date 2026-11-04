"""Reading Users.csv and matching each user to a DBLP author entity.

DBLP identifies authors by name strings, the input CSV by email; every output
of this system is keyed on email, so each user has to be resolved to the set of
DBLP aliases they publish under. That match is rarely exact, so each result
carries how confident it is (see the DISAM_* levels).
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from .constants import (
    DISAM_CERTAIN,
    DISAM_DOUBT,
    DISAM_FAILED,
    ROLES_COLUMN,
    USERS_CSV_DELIMITER,
)
from .dblp import canonical_pid, canonical_records

logger = logging.getLogger(__name__)

# Short tokens and stopwords are ignored when comparing affiliations
_MIN_ORG_TOKEN_LENGTH = 3
_ORG_STOPWORDS = frozenset({"of", "the", "and", "for", "in", "at", "a", "an"})

# Organisation/country tokens that must coincide to accept an affiliation
_MIN_AFFILIATION_TOKEN_OVERLAP = 3

# Exports disagree on whether roles are separated by ";" or ",".
_ROLE_SEPARATORS = re.compile(r"[;,]")


@dataclass(frozen=True)
class User:
    first_name: str
    last_name: str
    email: str
    organization: str
    country: str
    dblp_url: str
    dblp_pid: str | None
    gscholar_url: str
    semscholar_url: str
    openreview_id: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def identifying_urls(self) -> list[str]:
        """Profile URLs that identify this person on another site."""
        return [self.gscholar_url, self.semscholar_url, self.openreview_id]


class Disambiguation(NamedTuple):
    """The DBLP identity a user was matched to.

    `dblp_pid` is blank when no single entity could be pinned down, which is
    exactly the DISAM_FAILED case where several same-name entities are merged
    and no one pid applies.
    """

    aliases: set[str]
    level: str
    dblp_pid: str


class ResolvedUser(NamedTuple):
    """A Users.csv row together with the DBLP identity it was matched to."""

    user: User
    disambiguation: Disambiguation


class DblpIndexes(NamedTuple):
    """Inverted indexes over the homepage records; build once at startup."""

    by_url: dict[str, dict]  # Normalised URL -> homepage record
    by_name: dict[str, list[dict]]  # Lowercased name, suffix stripped -> records


def _pid_from_dblp_url(url: str) -> str | None:
    match = re.search(r"/pid/(.+?)(?:\.html)?$", (url or "").strip())
    return match.group(1) if match else None


def _has_reviewer_role(roles: str) -> bool:
    return "reviewer" in {
        role.strip().lower() for role in _ROLE_SEPARATORS.split(roles)
    }


# Row keyed by lowercased, trimmed column name; exports vary in casing
def _normalized(row: dict[str, str]) -> dict[str, str]:
    return {
        (key or "").strip().lower(): (value or "").strip() for key, value in row.items()
    }


def _user_from_fields(field: dict[str, str]) -> User:
    dblp_url = field.get("dblp url") or field.get("dblp_url") or field.get("dblp") or ""
    return User(
        first_name=field.get("first name", ""),
        last_name=field.get("last name", ""),
        email=field.get("email", ""),
        organization=field.get("organization", ""),
        country=field.get("country", ""),
        dblp_url=dblp_url,
        dblp_pid=_pid_from_dblp_url(dblp_url),
        gscholar_url=field.get("google scholar url", ""),
        semscholar_url=field.get("semantic scholar url", ""),
        openreview_id=field.get("openreview id", "") or field.get("openreview url", ""),
    )


def load_users_csv(csv_path: str, require_reviewer: bool = True) -> list[User]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        raw = fh.read()
    # The export comments out its own header row
    if raw.startswith("#"):
        raw = raw[1:]

    reader = csv.DictReader(raw.splitlines(), delimiter=USERS_CSV_DELIMITER)
    columns = {(name or "").strip().lower() for name in (reader.fieldnames or [])}
    # Without this the filter below matches nothing and the run reports 0 reviewers
    if require_reviewer and ROLES_COLUMN not in columns:
        raise SystemExit(
            f"{csv_path} has no '{ROLES_COLUMN}' column, so no row can be identified as a "
            f"reviewer. Columns found: {', '.join(sorted(columns)) or '(none)'}. "
            f"Check COI_USERS_CSV_DELIMITER (currently {USERS_CSV_DELIMITER!r})."
        )

    fields = (_normalized(row) for row in reader)
    return [
        _user_from_fields(field)
        for field in fields
        if not require_reviewer or _has_reviewer_role(field.get(ROLES_COLUMN, ""))
    ]


def build_dblp_indexes(homepages: dict[str, dict]) -> DblpIndexes:
    by_url: dict[str, dict] = {}
    by_name: dict[str, list[dict]] = defaultdict(list)
    url_collisions = 0
    for record in canonical_records(homepages):
        for url in record.get("urls", []):
            if not url:
                continue
            key = url.lower().rstrip("/")
            if key in by_url and by_url[key] is not record:
                url_collisions += 1
            by_url[key] = record
        base_names = {
            re.sub(r"\s+\d+$", "", alias).strip().lower()
            for alias in record.get("aliases", [])
        }
        for base_name in base_names:
            by_name[base_name].append(record)
    if url_collisions:
        # Two DBLP entities listing the same profile URL is a data problem worth seeing
        logger.warning(
            "%s identifying URL(s) appear on more than one DBLP homepage record; "
            "the last record read wins for those",
            f"{url_collisions:,}",
        )
    return DblpIndexes(by_url=by_url, by_name=dict(by_name))


# Significant words of an organisation or country name
# TODO: token overlap is not a good method, use string similarity instead
def _org_tokens(text: str) -> set[str]:
    return {
        word
        for word in re.split(r"[\s,;]+", text.lower())
        if len(word) >= _MIN_ORG_TOKEN_LENGTH and word not in _ORG_STOPWORDS
    }


# Whether any affiliation looks like the user's organisation or country
def _affiliation_matches(user: User, affiliations: list[str]) -> bool:
    user_tokens = _org_tokens(user.organization) | _org_tokens(user.country)
    if not user_tokens or not affiliations:
        return False
    return any(
        len(user_tokens & _org_tokens(affiliation)) >= _MIN_AFFILIATION_TOKEN_OVERLAP
        for affiliation in affiliations
    )


# Resolve an old-format DBLP person URL (/pers/hd/b/Doe_0001:John)
def _record_from_legacy_dblp_url(
    url: str, by_name: dict[str, list[dict]]
) -> dict | None:
    match = re.search(r"/pers/(?:[^/]+/)*([^/?#]+?)(?:\.html)?$", (url or "").strip())
    if not match:
        return None
    encoded = match.group(1)  # Example: "Doe_0001:John"
    separator = encoded.rfind(":")
    if separator < 0:
        return None

    last_name_part = encoded[:separator].replace("=", " ")
    first_name = encoded[separator + 1 :].replace("=", " ").strip()
    homonym_match = re.search(r"_(\d+)$", last_name_part)
    last_name = (
        last_name_part[: homonym_match.start()] if homonym_match else last_name_part
    ).strip()

    candidates = by_name.get(f"{first_name} {last_name}".lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates or homonym_match is None:
        return None

    suffix = f" {int(homonym_match.group(1)):04d}"
    return next(
        (
            record
            for record in candidates
            if any(
                alias.lower().endswith(suffix) for alias in record.get("aliases", [])
            )
        ),
        None,
    )


# Follows merged-profile pointers to the record holding the aliases
def _record_for_pid(pid: str, homepages: dict[str, dict]) -> dict | None:
    record = homepages.get(pid)
    seen: set[str] = set()
    while record is not None and record.get("crossref"):
        if record["pid"] in seen:
            logger.warning(
                "DBLP crossref cycle at pid %s; treating it as unresolved",
                record["pid"],
            )
            return None
        seen.add(record["pid"])
        record = homepages.get(canonical_pid(record["crossref"]))
    return record


# A Disambiguation for a record that carries aliases, else None
def _matched(record: dict | None, level: str) -> Disambiguation | None:
    if not record or not record.get("aliases"):
        return None
    return Disambiguation(set(record["aliases"]), level, record.get("pid", ""))


def _same_name_records(user: User, indexes: DblpIndexes) -> list[dict]:
    return indexes.by_name.get(user.full_name.lower(), [])


# The Users.csv row names a pid outright
def _by_dblp_pid(
    user: User, homepages: dict[str, dict], indexes: DblpIndexes
) -> Disambiguation | None:
    if not user.dblp_pid:
        return None
    return _matched(_record_for_pid(user.dblp_pid, homepages), DISAM_CERTAIN)


# An old-format /pers/ URL, resolvable only by name + homonym
def _by_legacy_dblp_url(
    user: User, homepages: dict[str, dict], indexes: DblpIndexes
) -> Disambiguation | None:
    if not user.dblp_url or user.dblp_pid:
        return None
    return _matched(
        _record_from_legacy_dblp_url(user.dblp_url, indexes.by_name), DISAM_CERTAIN
    )


# A Scholar/Semantic Scholar/OpenReview URL DBLP also lists
def _by_identifying_url(
    user: User, homepages: dict[str, dict], indexes: DblpIndexes
) -> Disambiguation | None:
    for url in user.identifying_urls:
        if not url:
            continue
        match = _matched(indexes.by_url.get(url.lower().rstrip("/")), DISAM_CERTAIN)
        if match:
            return match
    return None


# Exactly one DBLP entity carries this name
def _by_unique_name(
    user: User, homepages: dict[str, dict], indexes: DblpIndexes
) -> Disambiguation | None:
    records = _same_name_records(user, indexes)
    if len(records) != 1:
        return None
    return Disambiguation(
        set(records[0]["aliases"]), DISAM_DOUBT, records[0].get("pid", "")
    )


# Among same-name entities, one lists a matching organisation
def _by_affiliation(
    user: User, homepages: dict[str, dict], indexes: DblpIndexes
) -> Disambiguation | None:
    if not (user.organization or user.country):
        return None
    for record in _same_name_records(user, indexes):
        if _affiliation_matches(user, [a for a in record.get("affiliations", []) if a]):
            match = _matched(record, DISAM_DOUBT)
            if match:
                return match
    return None


_DISAMBIGUATION_LADDER = (
    _by_dblp_pid,
    _by_legacy_dblp_url,
    _by_identifying_url,
    _by_unique_name,
    _by_affiliation,
)


def disambiguate(
    user: User, homepages: dict[str, dict], indexes: DblpIndexes
) -> Disambiguation:
    """Match one user to a DBLP identity.

    1. DBLP URL (certain)
    2. DBLP /pers/ URL (certain)
    3. Scholar / Semantic Scholar / OpenReview URL (certain)
    4. Globally unique name (doubt)
    5. Organisation or country matches an affiliation (doubt)
    6. No singular entity found (failed)
    """
    for rung in _DISAMBIGUATION_LADDER:
        match = rung(user, homepages, indexes)
        if match:
            return match

    return Disambiguation({user.full_name}, DISAM_FAILED, "")


def resolve_users(
    homepages: dict[str, dict], users_csv_path: str
) -> list[ResolvedUser]:
    """Match every user in the CSV to a DBLP identity.

    Covers all roles, not just reviewers, since a COI can involve any author in
    the input. Users without an email are skipped: every output is keyed on
    email, so they could not be reported either way.
    """
    indexes = build_dblp_indexes(homepages)
    return [
        ResolvedUser(user, disambiguate(user, homepages, indexes))
        for user in load_users_csv(users_csv_path, require_reviewer=False)
        if user.email
    ]


def unresolved_users(resolved: list[ResolvedUser]) -> list[ResolvedUser]:
    """The users whose DBLP identity could not be pinned down."""
    return [entry for entry in resolved if entry.disambiguation.level == DISAM_FAILED]


def build_author_email_index(
    resolved: list[ResolvedUser],
) -> dict[str, tuple[str, str, str]]:
    """Every DBLP alias a user may publish under -> (email, pid, disambiguation).

    Turns the author-name nodes of the COI graph back into the email addresses
    the output CSVs are keyed on.

    DISAM_FAILED users are left out: their "alias" is just their own full name,
    not a real DBLP identity. They are listed in the unresolved-users report.

    An alias can only map to one user. When two users (homonyms) claim the same
    alias, the last one seen wins and the collision is logged.
    """
    index: dict[str, tuple[str, str, str]] = {}
    collisions = 0
    for entry in resolved:
        if entry.disambiguation.level == DISAM_FAILED:
            continue
        value = (
            entry.user.email,
            entry.disambiguation.dblp_pid,
            entry.disambiguation.level,
        )
        for alias in entry.disambiguation.aliases:
            if index.get(alias, value)[0] != value[0]:
                collisions += 1
            index[alias] = value
    if collisions:
        logger.warning(
            "%s DBLP alias(es) are claimed by more than one user; the last user seen wins "
            "and every conflict through those names is attributed to them",
            f"{collisions:,}",
        )
    return index
