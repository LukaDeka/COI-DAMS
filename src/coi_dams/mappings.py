"""Small lookup tables built on the driver and handed to Spark.

Both tables here are derived from the homepage index and Users.csv, so they are
built in plain Python and then have to become DataFrames
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from . import webhdfs
from .dblp import canonical_homepage_records

logger = logging.getLogger(__name__)

# (local_tmp_path, hdfs_uri, webhdfs_base), hdfs_uri is the upload target and what Spark reads back
HdfsUpload = tuple[str, str, str]


# Builds a string DataFrame, going through Parquet with hdfs_upload so no Python worker is needed
def _table_from_rows(
    spark: SparkSession,
    columns: list[str],
    # Sequence because list is invariant
    rows: Sequence[tuple[str, ...]],
    hdfs_upload: HdfsUpload | None,
) -> DataFrame:
    if hdfs_upload is None:
        return spark.createDataFrame(
            rows, schema=", ".join(f"{c} string" for c in columns)
        )

    local_tmp_path, hdfs_uri, webhdfs_base = hdfs_upload
    table = pa.Table.from_pylist(
        [dict(zip(columns, row, strict=True)) for row in rows],
        schema=pa.schema([(column, pa.string()) for column in columns]),
    )
    pq.write_table(table, local_tmp_path)
    webhdfs.upload_file(local_tmp_path, webhdfs.uri_to_path(hdfs_uri), webhdfs_base)
    return spark.read.parquet(hdfs_uri)


def build_alias_pid_df(
    spark: SparkSession,
    homepages_parquet: str,
    hdfs_upload: HdfsUpload | None = None,
) -> DataFrame:
    """Every DBLP alias with exactly one pid -> that pid, for any author in the dump.

    Reads the homepage index in batches.
    Covers all authors, because the intermediate authors
    inside a COI evidence path don't have to be users.
    """
    pid_by_alias: dict[str, str] = {}
    ambiguous: set[str] = set()
    for record in canonical_homepage_records(homepages_parquet):
        pid = record.get("pid", "")
        for alias in record.get("aliases", []):
            if pid_by_alias.setdefault(alias, pid) != pid:
                ambiguous.add(alias)
    if ambiguous:
        logger.warning(
            "%s DBLP alias(es) map to more than one pid; they are left out, so each "
            "stays a node of its own in the COI graph",
            f"{len(ambiguous):,}",
        )
    rows = [
        (alias, pid) for alias, pid in pid_by_alias.items() if alias not in ambiguous
    ]
    return _table_from_rows(spark, ["author_name", "dblp_id"], rows, hdfs_upload)


def build_email_mapping_df(
    spark: SparkSession,
    email_index: dict[str, tuple[str, str, str]],
    hdfs_upload: HdfsUpload | None = None,
) -> DataFrame:
    """Known-user alias -> (email, dblp pid, disambiguation level)."""
    rows = [
        (name, email, pid, level) for name, (email, pid, level) in email_index.items()
    ]
    return _table_from_rows(
        spark, ["author_name", "email", "dblp_id", "disam_level"], rows, hdfs_upload
    )


# A known user's COI graph node: one id per email however many aliases they have
def _user_node(email: Column) -> Column:
    return F.concat(F.lit("u:"), email)


def build_alias_node_df(alias_pid_df: DataFrame, mapping: DataFrame) -> DataFrame:
    """Every alias -> the id of the person it stands for in the COI graph.

    A known user's alias, even one shared with another pid, and any alias of a
    known user's pid is "u:" + email. Another alias with a single pid is
    "p:" + pid. The rest (ambiguous, unknown, synthetic ids) have no row.
    """
    by_email = mapping.select(
        "author_name", _user_node(F.col("email")).alias("node_id")
    )
    user_by_pid = mapping.groupBy("dblp_id").agg(F.min("email").alias("email"))
    by_pid = (
        alias_pid_df.join(F.broadcast(user_by_pid), on="dblp_id", how="left")
        .select(
            "author_name",
            F.coalesce(
                _user_node(F.col("email")), F.concat(F.lit("p:"), F.col("dblp_id"))
            ).alias("node_id"),
        )
        .join(F.broadcast(by_email.select("author_name")), "author_name", "left_anti")
    )
    return by_pid.unionByName(by_email)


def build_node_mapping_df(mapping: DataFrame) -> DataFrame:
    """Known users keyed on their COI graph node, one row per user."""
    return (
        mapping.groupBy("email")
        .agg(F.min(F.struct("author_name", "dblp_id", "disam_level")).alias("user"))
        .select(
            _user_node(F.col("email")).alias("node_id"),
            "user.author_name",
            "email",
            "user.dblp_id",
            "user.disam_level",
        )
    )
