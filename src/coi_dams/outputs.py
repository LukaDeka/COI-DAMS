"""Writing the pipeline's result DataFrames out as single CSV files.

Every writer takes an optional `hdfs_tmp`: on a cluster the executors write the
partitions, so the output lands on HDFS and is pulled back to local disk before
concat.
"""

from __future__ import annotations

import csv
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from . import webhdfs
from .constants import DISAM_CERTAIN, DISAM_DOUBT, DISAM_FAILED
from .spark_helpers import lit_map
from .users import ResolvedUser

logger = logging.getLogger(__name__)

# (hdfs_dir_uri, webhdfs_base) for a cluster run, None for local mode.
HdfsTmp = tuple[str, str]

# Ranked worst to best, a pair's confidence is its weaker side
_DISAM_RANK = {DISAM_FAILED: 0, DISAM_DOUBT: 1, DISAM_CERTAIN: 2}


# Writes CSV parts and appends them into one file
def _write_csv(
    df: DataFrame,
    out_path: Path,
    hdfs_tmp: HdfsTmp | None,
    header: str | None,
    separator: str = ",",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    local_tmp_dir = str(out_path.with_suffix("._spark_out"))
    (
        df.write.mode("overwrite")
        .option("header", "false")
        .option("sep", separator)
        .option("escape", '"')  # Escape CSV with double quotes
        .csv(hdfs_tmp[0] if hdfs_tmp else local_tmp_dir)
    )
    try:
        if hdfs_tmp:
            webhdfs.download_dir(hdfs_tmp[0], local_tmp_dir, hdfs_tmp[1])

        with open(out_path, "w", encoding="utf-8") as out_file:
            if header is not None:
                out_file.write(header + "\n")
            for part in sorted(Path(local_tmp_dir).glob("part-*.csv")):
                with open(part, encoding="utf-8") as part_file:
                    shutil.copyfileobj(part_file, out_file)
    finally:
        # The staging directory holds a second full copy of the output, so clean up on failure too
        shutil.rmtree(local_tmp_dir, ignore_errors=True)
        if hdfs_tmp:
            try:
                webhdfs.delete_dir(hdfs_tmp[0], hdfs_tmp[1])
            except OSError as error:
                logger.warning(
                    "Could not remove HDFS staging dir %s: %s", hdfs_tmp[0], error
                )


def write_coi_csv(
    df: DataFrame, output_path: str, threshold: int, hdfs_tmp: HdfsTmp | None = None
) -> None:
    """Main output: tab-separated email pairs scoring at or above `threshold`.

    Inclusive, since a depth-3 pair with one evidence path scores exactly the
    default threshold.
    """
    above_threshold = (
        df.filter(F.col("coi_score") >= threshold)
        .select("email_a", "email_b")
        .orderBy("email_a", "email_b")
    )
    _write_csv(
        above_threshold, Path(output_path), hdfs_tmp, header=None, separator="\t"
    )


# Drops DBLP's homonym suffix ("Fatma Ozcan 0001") so it isn't taken for the surname
def _strip_homonym_suffix(column: str) -> Column:
    return F.regexp_replace(F.col(column), r"\s+\d{4}$", "")


# Splits "First Middle Last" into (given names, surname) without a UDF
def _name_and_surname(full_name: Column) -> tuple[Column, Column]:
    parts = F.split(full_name, " ")
    surname = F.element_at(parts, -1)
    given = F.array_join(F.slice(parts, 1, F.size(parts) - 1), " ")
    return given, surname


# Add the per-side name columns and the pair's overall confidence
def _with_presentation_columns(df: DataFrame) -> DataFrame:
    clean_a = _strip_homonym_suffix("author_a")
    clean_b = _strip_homonym_suffix("author_b")
    name_a, surname_a = _name_and_surname(clean_a)
    name_b, surname_b = _name_and_surname(clean_b)

    # A pair is only as trustworthy as its weaker side
    rank_of_level = lit_map(_DISAM_RANK)
    level_of_rank = lit_map({rank: level for level, rank in _DISAM_RANK.items()})
    weaker_level = level_of_rank[
        F.least(
            rank_of_level[F.col("disam_level_a")], rank_of_level[F.col("disam_level_b")]
        )
    ]

    return (
        df.withColumn("name_a", name_a)
        .withColumn("surname_a", surname_a)
        .withColumn("full_name_a", clean_a)
        .withColumn("name_b", name_b)
        .withColumn("surname_b", surname_b)
        .withColumn("full_name_b", clean_b)
        .withColumn("disambiguation_total", weaker_level)
        .select(
            "email_a",
            "name_a",
            "surname_a",
            "full_name_a",
            "dblp_id_a",
            "disam_level_a",
            "email_b",
            "name_b",
            "surname_b",
            "full_name_b",
            "dblp_id_b",
            "disam_level_b",
            "disambiguation_total",
            "coi_score",
            "depth",
            "coi_type",
            "evidence_path_count",
            "evidence_papers",
        )
    )


def write_analysis_csv(
    df: DataFrame, output_path: str, hdfs_tmp: HdfsTmp | None = None
) -> None:
    """Analysis output: every scored pair, regardless of threshold.

    Carries names and email per side, the evidence chains, and how confidently
    each side was disambiguated. Sorted here because this is the last stage
    before the write.
    """
    enriched = _with_presentation_columns(df).orderBy(
        F.col("coi_score").desc(), "email_a", "email_b"
    )
    _write_csv(enriched, Path(output_path), hdfs_tmp, header=",".join(enriched.columns))


_UNRESOLVED_COLUMNS = [
    "email",
    "full_name",
    "organization",
    "country",
    "disambiguation",
    "reason",
]
_UNRESOLVED_REASON = (
    "several DBLP entities share this name and none could be picked - check manually"
)


def write_unresolved_users_csv(users: Sequence[ResolvedUser], output_path: str) -> None:
    """Report the users left out of the analysis because their identity is unclear.

    Without this file such a user is absent from the COI output, which reads as
    "has no conflicts" instead of "could not be checked".

    Plain Python since it is a driver-side list of a few hundred rows at most.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(_UNRESOLVED_COLUMNS)
        writer.writerows(
            [
                entry.user.email,
                entry.user.full_name,
                entry.user.organization,
                entry.user.country,
                entry.disambiguation.level,
                _UNRESOLVED_REASON,
            ]
            for entry in users
        )
