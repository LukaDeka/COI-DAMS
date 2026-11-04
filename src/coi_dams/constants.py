"""Tunable values, shared schemas and lookups"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa

DOTENV_PATH = ".env"

_QUOTE_CHARACTERS = ("'", '"')


# Strips surrounding quotes from env values, e.g. `KEY="5"` to `5`
def _unquoted(value: str) -> str:
    return value[1:-1] if _is_quoted(value) else value


def _is_quoted(value: str) -> bool:
    return len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTE_CHARACTERS


# Yields key/value pairs. A quoted value may span lines, an unclosed one ends at EOF
def _dotenv_entries(lines: list[str]) -> Iterator[tuple[str, str]]:
    key: str | None = None
    collected: list[str] = []
    quote = ""
    for line in lines:
        if key is not None:  # Multi-line value
            body = line.rstrip()
            if body.endswith(quote):
                yield key, "\n".join([*collected, body[:-1]])
                key = None
            else:
                collected.append(body)
            continue

        stripped = line.strip()
        if (
            not stripped or stripped.startswith("#") or "=" not in stripped
        ):  # Ignore comments or unfinished lines
            continue
        name, _, value = stripped.partition("=")
        value = value.strip()
        if value[:1] in _QUOTE_CHARACTERS and not _is_quoted(value):
            key, quote, collected = name.strip(), value[0], [value[1:]]
            continue
        yield name.strip(), _unquoted(value)
    if key is not None:
        yield key, "\n".join(collected)


# Populate os.environ from the env file
def _load_dotenv(path: str = DOTENV_PATH) -> None:
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for key, value in _dotenv_entries(dotenv.read_text(encoding="utf-8").splitlines()):
        os.environ.setdefault(key, value)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(_unquoted(raw.strip()))
    except ValueError:
        raise SystemExit(
            f"{name} must be an integer, got {raw!r} - check the environment or {DOTENV_PATH}"
        ) from None


_load_dotenv()


# Publications older than this are discarded for computation since they're too old
OLDEST_CUTOFF_YEAR = env_int("COI_OLDEST_CUTOFF_YEAR", 2016)

# Publication records buffered before each Parquet write while streaming the DBLP XML
XML_PARSE_CHUNK_SIZE = env_int("COI_CHUNK_SIZE", 10_000)

# Worker processes used to parse the Semantic Scholar authors/papers dumps in parallel
S2_PARSE_WORKERS = env_int("COI_S2_PARSE_WORKERS", 4)

# Column separator of the input Users.csv
USERS_CSV_DELIMITER = os.environ.get("COI_USERS_CSV_DELIMITER", "\t")


# COI scoring

# Score given to a pair per evidence path of set depth
DEFAULT_WEIGHTS = {
    "depth_1": env_int("COI_WEIGHT_DEPTH_1", 100),
    "depth_2": env_int("COI_WEIGHT_DEPTH_2", 20),
    "depth_3": env_int("COI_WEIGHT_DEPTH_3", 5),
    "depth_4": env_int("COI_WEIGHT_DEPTH_4", 2),
}

# Default for --threshold. Pairs scoring below it only appear in the analysis CSV. The comparison is inclusive
COI_ACCEPT_THRESHOLD = env_int("COI_ACCEPT_THRESHOLD", 5)

WEIGHT_KEY_PREFIX = "depth_"


def weight_depth(key: str) -> int:
    suffix = key[len(WEIGHT_KEY_PREFIX) :]
    if not key.startswith(WEIGHT_KEY_PREFIX) or not suffix.isdigit():
        raise ValueError(
            f"weight key must look like '{WEIGHT_KEY_PREFIX}<n>', got {key!r}"
        )
    return int(suffix)


COI_TYPE_MAP = {
    1: "direct_coauthor",
    2: "indirect_depth_2",
    3: "indirect_depth_3",
    4: "submarine_depth_4",
}

# Reviewer qualification

# Minimum number of QUALIFYING_VENUES papers for a reviewer to count as qualified
REVIEWER_MIN_PAPERS = env_int("COI_REVIEWER_MIN_PAPERS", 2)

# How confidently a Users.csv row was matched to a DBLP entity.
DISAM_CERTAIN = "certain"  # Matched on a DBLP pid or an identifying URL
DISAM_DOUBT = "doubt"  # Matched on a unique name or the organisation/country heuristic
DISAM_FAILED = "failed"  # Unresolved, excluded from computation

# Column of Users.csv that carries the semicolon/comma-separated role list.
ROLES_COLUMN = "roles"


# DBLP vocabulary and schemas

DBLP_SCHEMA = pa.schema(
    [
        pa.field("key", pa.string()),
        pa.field("title", pa.string()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("year", pa.int16()),
        pa.field("venue", pa.string()),
        pa.field("pub_type", pa.string()),
        pa.field("doi", pa.string()),
    ]
)

HOMEPAGE_SCHEMA = pa.schema(
    [
        pa.field("pid", pa.string()),
        # Set on a merged profile's secondary record, pointing at the primary
        pa.field("crossref", pa.string()),
        pa.field("aliases", pa.list_(pa.string())),
        pa.field("affiliations", pa.list_(pa.string())),
        pa.field("urls", pa.list_(pa.string())),
    ]
)

PUB_TAGS = frozenset(
    [
        "article",
        "inproceedings",
        "proceedings",
        "book",
        "incollection",
        "phdthesis",
        "mastersthesis",
    ]
)

# Used when formatting an evidence path like "Conference(A/id, B/id, paperkey)"
PUB_TYPE_LABELS = {
    "article": "Journal",
    "inproceedings": "Conference",
    "proceedings": "Proceedings",
    "book": "Book",
    "incollection": "Incollection",
    "phdthesis": "Thesis",
    "mastersthesis": "Thesis",
}
