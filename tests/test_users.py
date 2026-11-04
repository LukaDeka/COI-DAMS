from __future__ import annotations

import logging

import pytest

from coi_dams.constants import (
    DISAM_CERTAIN,
    DISAM_DOUBT,
    DISAM_FAILED,
    USERS_CSV_DELIMITER,
)
from coi_dams.users import (
    Disambiguation,
    ResolvedUser,
    User,
    _affiliation_matches,
    _has_reviewer_role,
    _pid_from_dblp_url,
    _record_for_pid,
    _record_from_legacy_dblp_url,
    build_author_email_index,
    build_dblp_indexes,
    disambiguate,
    load_users_csv,
)


def homepage(pid, aliases=(), urls=(), affiliations=(), crossref=None):
    return {
        "pid": pid,
        "crossref": crossref,
        "aliases": list(aliases),
        "affiliations": list(affiliations),
        "urls": list(urls),
    }


def user(**overrides) -> User:
    fields = {
        "first_name": "Jane",
        "last_name": "Doe",
        "email": "jane@example.org",
        "organization": "",
        "country": "",
        "dblp_url": "",
        "dblp_pid": None,
        "gscholar_url": "",
        "semscholar_url": "",
        "openreview_id": "",
    }
    fields.update(overrides)
    return User(**fields)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://dblp.org/pid/12/3456", "12/3456"),
        ("https://dblp.org/pid/12/3456.html", "12/3456"),
        ("  https://dblp.org/pid/l/Luka.html  ", "l/Luka"),
        ("https://dblp.org/pers/hd/d/Doe:Jane", None),
        ("", None),
        ("not a url", None),
    ],
)
def test_pid_from_dblp_url(url, expected):
    assert _pid_from_dblp_url(url) == expected


def test_legacy_url_unique():
    record = homepage("d/JaneDoe", aliases=["Jane Doe"])
    by_name = {"jane doe": [record]}
    assert (
        _record_from_legacy_dblp_url("https://dblp.org/pers/hd/d/Doe:Jane", by_name)
        is record
    )


def test_legacy_url_homonym():
    first = homepage("d/JaneDoe1", aliases=["Jane Doe 0001"])
    second = homepage("d/JaneDoe2", aliases=["Jane Doe 0002"])
    by_name = {"jane doe": [first, second]}
    resolved = _record_from_legacy_dblp_url(
        "https://dblp.org/pers/hd/d/Doe_0002:Jane", by_name
    )
    assert resolved is second


def test_legacy_url_ambiguous():
    by_name = {
        "jane doe": [
            homepage("a", aliases=["Jane Doe"]),
            homepage("b", aliases=["Jane Doe"]),
        ]
    }
    assert (
        _record_from_legacy_dblp_url("https://dblp.org/pers/hd/d/Doe:Jane", by_name)
        is None
    )


@pytest.mark.parametrize("url", ["", "https://dblp.org/pers/hd/d/NoColonHere"])
def test_legacy_url_unparsable(url):
    assert _record_from_legacy_dblp_url(url, {}) is None


def test_record_follows_pointer():
    """The secondary pid of a merged profile must still resolve."""
    target = homepage("t/Target", aliases=["Jane Doe", "J. Doe"])
    pointer = homepage("p/Pointer", crossref="homepages/t/Target")
    homepages = {"t/Target": target, "p/Pointer": pointer}
    assert _record_for_pid("p/Pointer", homepages) is target


def test_record_multi_hop():
    target = homepage("c", aliases=["Jane Doe"])
    homepages = {
        "a": homepage("a", crossref="homepages/b"),
        "b": homepage("b", crossref="homepages/c"),
        "c": target,
    }
    assert _record_for_pid("a", homepages) is target


def test_record_cycle(caplog):
    homepages = {
        "a": homepage("a", crossref="homepages/b"),
        "b": homepage("b", crossref="homepages/a"),
    }
    with caplog.at_level(logging.WARNING):
        assert _record_for_pid("a", homepages) is None
    assert "cycle" in caplog.text


def test_affiliation_needs_three_tokens():
    jane = user(organization="University of California, Berkeley", country="USA")
    assert _affiliation_matches(jane, ["University of California, Berkeley, USA"])
    # Two shared tokens is below the threshold
    assert not _affiliation_matches(jane, ["University of California"])


def test_disambiguate_prefers_the_pid(caplog):
    record = homepage("12/3456", aliases=["Jane Doe", "Jane A. Doe"])
    homepages = {"12/3456": record}
    match = disambiguate(
        user(dblp_url="https://dblp.org/pid/12/3456", dblp_pid="12/3456"),
        homepages,
        build_dblp_indexes(homepages),
    )
    assert match == Disambiguation(
        {"Jane Doe", "Jane A. Doe"}, DISAM_CERTAIN, "12/3456"
    )


def test_disambiguate_by_url():
    record = homepage(
        "x",
        aliases=["Jane Doe"],
        urls=["https://scholar.google.com/citations?user=Q1/"],
    )
    homepages = {"x": record}
    match = disambiguate(
        user(
            first_name="Not",
            last_name="Matching",
            gscholar_url="https://scholar.google.com/citations?user=Q1",
        ),
        homepages,
        build_dblp_indexes(homepages),
    )
    assert match.level == DISAM_CERTAIN
    assert match.dblp_pid == "x"


def test_disambiguate_unique_name():
    homepages = {"x": homepage("x", aliases=["Jane Doe"])}
    match = disambiguate(user(), homepages, build_dblp_indexes(homepages))
    assert match.level == DISAM_DOUBT


def test_disambiguate_by_affiliation():
    homepages = {
        "a": homepage(
            "a",
            aliases=["Jane Doe"],
            affiliations=["University of California, Berkeley"],
        ),
        "b": homepage("b", aliases=["Jane Doe"], affiliations=["Some Other Place"]),
    }
    match = disambiguate(
        user(organization="University of California, Berkeley"),
        homepages,
        build_dblp_indexes(homepages),
    )
    assert match.level == DISAM_DOUBT
    assert match.dblp_pid == "a"


def test_disambiguate_same_name_fails():
    homepages = {
        "a": homepage("a", aliases=["Jane Doe"]),
        "b": homepage("b", aliases=["Jane Doe 0002"]),
    }
    # Both records index under "jane doe", so neither is picked and only the user's own name is returned
    match = disambiguate(user(), homepages, build_dblp_indexes(homepages))
    assert match.level == DISAM_FAILED
    assert match.dblp_pid == ""
    assert match.aliases == {"Jane Doe"}


def test_merged_profile_unique_name():
    """Both halves of a merged profile must not look like two same-name people."""
    target = homepage("t", aliases=["Jane Doe", "J. Doe"])
    pointer = homepage("p", aliases=["J. Doe"], crossref="homepages/t")
    homepages = {"t": target, "p": pointer}
    match = disambiguate(user(), homepages, build_dblp_indexes(homepages))
    assert match.level == DISAM_DOUBT
    assert match.dblp_pid == "t"


def test_unique_name_with_homonym_suffixed_alias():
    homepages = {"t": homepage("t", aliases=["Jane Doe", "Jane Doe 0001"])}
    match = disambiguate(user(), homepages, build_dblp_indexes(homepages))
    assert match.level == DISAM_DOUBT
    assert match.dblp_pid == "t"


def test_indexes_warn_on_shared_url(caplog):
    homepages = {
        "a": homepage("a", aliases=["A"], urls=["https://example.org/p"]),
        "b": homepage("b", aliases=["B"], urls=["https://example.org/p/"]),
    }
    with caplog.at_level(logging.WARNING):
        build_dblp_indexes(homepages)
    assert "more than one DBLP homepage record" in caplog.text


def write_users_csv(tmp_path, header, *rows, comment_header=True):
    delimiter = USERS_CSV_DELIMITER
    lines = [("#" if comment_header else "") + delimiter.join(header)]
    lines += [delimiter.join(row) for row in rows]
    path = tmp_path / "Users.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_users_csv_commented_header(tmp_path):
    path = write_users_csv(
        tmp_path,
        ["First Name", "Last Name", "Email", "Roles", "DBLP URL"],
        [
            "Jane",
            "Doe",
            "jane@example.org",
            "reviewer;author",
            "https://dblp.org/pid/12/3456",
        ],
    )
    (loaded,) = load_users_csv(path)
    assert loaded.full_name == "Jane Doe"
    assert loaded.email == "jane@example.org"
    assert loaded.dblp_pid == "12/3456"


@pytest.mark.parametrize(
    "roles", ["reviewer", "Reviewer; author", "author,reviewer", " reviewer "]
)
def test_reviewer_role_separators(roles):
    assert _has_reviewer_role(roles)


@pytest.mark.parametrize("roles", ["", "author", "reviewers", "author;chair"])
def test_reviewer_role_rejects(roles):
    assert not _has_reviewer_role(roles)


def test_users_csv_no_roles_column(tmp_path):
    """A missing roles column is an error, not "0 reviewers"."""
    path = write_users_csv(
        tmp_path,
        ["First Name", "Last Name", "Email"],
        ["Jane", "Doe", "jane@example.org"],
    )
    with pytest.raises(SystemExit) as error:
        load_users_csv(path)
    assert "roles" in str(error.value)


def resolved(email, aliases, level=DISAM_CERTAIN, pid="p"):
    return ResolvedUser(user(email=email), Disambiguation(set(aliases), level, pid))


def test_email_index_maps_aliases():
    index = build_author_email_index(
        [resolved("jane@example.org", ["Jane Doe", "J. Doe"])]
    )
    assert index == {
        "Jane Doe": ("jane@example.org", "p", DISAM_CERTAIN),
        "J. Doe": ("jane@example.org", "p", DISAM_CERTAIN),
    }


def test_email_index_skips_failed():
    entries = [
        resolved("jane@example.org", ["Jane Doe"]),
        resolved("john@example.org", ["John Roe"], level=DISAM_FAILED, pid=""),
    ]
    assert set(build_author_email_index(entries)) == {"Jane Doe"}


def test_email_index_alias_clash(caplog):
    entries = [
        resolved("jane@example.org", ["Jane Doe"]),
        resolved("other@example.org", ["Jane Doe"]),
    ]
    with caplog.at_level(logging.WARNING):
        index = build_author_email_index(entries)
    assert index["Jane Doe"][0] == "other@example.org"  # Last one wins
    assert "claimed by more than one user" in caplog.text
