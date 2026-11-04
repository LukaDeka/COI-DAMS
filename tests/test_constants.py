from __future__ import annotations

import os

import pytest

from coi_dams.constants import _load_dotenv, _unquoted, env_int, weight_depth


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"5"', "5"),
        ("'5'", "5"),
        ("5", "5"),
        ('"', '"'),  # A lone quote is not a pair
        ('"unbalanced', '"unbalanced'),
        ("", ""),
    ],
)
def test_unquoted_matching_pairs(raw, expected):
    assert _unquoted(raw) == expected


def test_dotenv_quotes_and_existing_env(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\n"
        "\n"
        'COI_TEST_QUOTED="7"\n'
        "COI_TEST_PLAIN=8\n"
        "COI_TEST_ALREADY_SET=9\n"
        "not a pair\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("COI_TEST_QUOTED", raising=False)
    monkeypatch.delenv("COI_TEST_PLAIN", raising=False)
    monkeypatch.setenv("COI_TEST_ALREADY_SET", "11")

    _load_dotenv(str(dotenv))

    assert env_int("COI_TEST_QUOTED", 0) == 7
    assert env_int("COI_TEST_PLAIN", 0) == 8
    # An explicit environment variable beats the checked-in file
    assert env_int("COI_TEST_ALREADY_SET", 0) == 11


def test_dotenv_missing_file(tmp_path):
    _load_dotenv(str(tmp_path / "absent"))  # Must not raise


def testenv_int_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("COI_TEST_MISSING", raising=False)
    assert env_int("COI_TEST_MISSING", 42) == 42


def testenv_int_names_the_variable_it_could_not_parse(monkeypatch):
    monkeypatch.setenv("COI_TEST_BAD", "not-a-number")
    with pytest.raises(SystemExit) as error:
        env_int("COI_TEST_BAD", 1)
    assert "COI_TEST_BAD" in str(error.value)


@pytest.mark.parametrize(("key", "depth"), [("depth_1", 1), ("depth_10", 10)])
def test_weight_depth_reads_the_suffix(key, depth):
    assert weight_depth(key) == depth


@pytest.mark.parametrize("key", ["depth1", "depth_", "depth_x", "score_2", ""])
def test_weight_depth_rejects(key):
    with pytest.raises(ValueError):
        weight_depth(key)


def test_dotenv_multiline(tmp_path, monkeypatch):
    """A long list is only readable one item per line, so quoted values may wrap."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'COI_TEST_MULTILINE="\n'
        "  A: one, two\n"
        "  B: three\n"
        '"\n'
        "COI_TEST_AFTER=5\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("COI_TEST_MULTILINE", raising=False)
    monkeypatch.delenv("COI_TEST_AFTER", raising=False)

    _load_dotenv(str(dotenv))

    assert os.environ["COI_TEST_MULTILINE"].splitlines() == [
        "",
        "  A: one, two",
        "  B: three",
    ]
    # Parsing resumes after the closing quote
    assert env_int("COI_TEST_AFTER", 0) == 5


def test_dotenv_unclosed_quote(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text('COI_TEST_UNCLOSED="a\nb\n', encoding="utf-8")
    monkeypatch.delenv("COI_TEST_UNCLOSED", raising=False)

    _load_dotenv(str(dotenv))  # Must not raise

    assert os.environ["COI_TEST_UNCLOSED"] == "a\nb"
