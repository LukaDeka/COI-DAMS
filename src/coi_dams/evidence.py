"""Rendering of COI evidence paths into their human-readable form."""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .constants import PUB_TYPE_LABELS
from .spark_helpers import lit_map

HOP_SEPARATOR = "+"
PATH_SEPARATOR = ";"

# The pair columns every COI row is keyed on when evidence is rendered
PAIR_KEY_COLUMNS = ("email_a", "email_b")


# One row per hop, with indices because joins don't keep order
def _hops_of_each_path(df: DataFrame, keys: list[str]) -> DataFrame:
    exploded_paths = df.select(*keys, F.posexplode("paths").alias("path_idx", "path"))
    return exploded_paths.select(
        *keys,
        "path_idx",
        F.posexplode("path").alias("hop_idx", "hop"),
    ).select(
        *keys,
        "path_idx",
        "hop_idx",
        F.col("hop.from_author").alias("from_author"),
        F.col("hop.to_author").alias("to_author"),
        F.col("hop.paper").alias("paper"),
        F.col("hop.pub_type").alias("pub_type"),
    )


# Renders hops as "Conference(A/idA, B/idB, paperkey)".
def _with_hop_strings(hops: DataFrame, alias_pid_df: DataFrame) -> DataFrame:
    from_pids = alias_pid_df.withColumnRenamed("dblp_id", "from_id").alias("m1")
    to_pids = alias_pid_df.withColumnRenamed("dblp_id", "to_id").alias("m2")
    label_of_pub_type = lit_map(PUB_TYPE_LABELS)
    return (
        hops.join(from_pids, hops["from_author"] == F.col("m1.author_name"), "left")
        .join(to_pids, hops["to_author"] == F.col("m2.author_name"), "left")
        .withColumn(
            "hop_str",
            F.concat(
                F.coalesce(label_of_pub_type[F.col("pub_type")], F.col("pub_type")),
                F.lit("("),
                F.col("from_author"),
                F.lit("/"),
                F.coalesce(F.col("from_id"), F.lit("")),
                F.lit(", "),
                F.col("to_author"),
                F.lit("/"),
                F.coalesce(F.col("to_id"), F.lit("")),
                F.lit(", "),
                F.col("paper"),
                F.lit(")"),
            ),
        )
    )


# Collect `value` per group, restore `index` order, and join with `separator`
def _joined_in_index_order(
    df: DataFrame,
    group_by: list[str],
    index: str,
    value: str,
    separator: str,
    output: str,
) -> DataFrame:
    return (
        df.groupBy(*group_by)
        .agg(
            F.array_sort(F.collect_list(F.struct(F.col(index), F.col(value)))).alias(
                "ordered"
            )
        )
        .withColumn(
            output,
            F.concat_ws(
                separator, F.transform(F.col("ordered"), lambda row: row[value])
            ),
        )
    )


def format_paths(
    df: DataFrame,
    alias_pid_df: DataFrame,
    key_columns: Sequence[str] = PAIR_KEY_COLUMNS,
) -> DataFrame:
    keys = list(key_columns)
    hops = _with_hop_strings(
        _hops_of_each_path(df.select(*keys, "paths"), keys), alias_pid_df
    )
    paths = _joined_in_index_order(
        hops, [*keys, "path_idx"], "hop_idx", "hop_str", HOP_SEPARATOR, "path_str"
    )
    rendered = _joined_in_index_order(
        paths, keys, "path_idx", "path_str", PATH_SEPARATOR, "evidence_papers"
    ).select(*keys, "evidence_papers")
    # Left join, pairs deeper than the evidence goes have null `paths` but still count
    return df.drop("paths").join(rendered, on=keys, how="left")
