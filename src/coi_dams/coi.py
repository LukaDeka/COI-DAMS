"""COI detection over the DBLP co-authorship graph.

A COI pair is two authors connected by a chain of co-authorship. BFS runs from
every source author at once, and a final aggregation merges the results.
Workers need no coordination

Authors are canonicalised before the BFS, so one person is one node however many
aliases they publish under. The BFS walks person-to-person edges and counts the
shortest chains of people instead of carrying them.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from .constants import COI_TYPE_MAP, env_int, weight_depth
from .spark_helpers import hop_struct, lit_map

logger = logging.getLogger(__name__)

# Papers with many authors are dropped, they aren't strong COIs and slow the computation
MAX_COAUTHORS_PER_PAPER = env_int("COI_MAX_COAUTHORS_PER_PAPER", 50)


def _checkpoint_path(checkpoint_dir: str, name: str) -> str:
    return f"{checkpoint_dir.rstrip('/')}/{name}"


def _materialize(spark: SparkSession, df: DataFrame, path: str) -> DataFrame:
    df.write.mode("overwrite").parquet(path)
    return spark.read.parquet(path)


# Bipartite paper/author rows, `author` is the node id and `name` the name as printed
def _read_paper_author_edges(
    spark: SparkSession, parquet_path: str, alias_node_df: DataFrame | None
) -> DataFrame:
    papers = (
        spark.read.parquet(parquet_path)
        .filter(F.size("authors") >= 2)
        .filter(F.size("authors") <= MAX_COAUTHORS_PER_PAPER)
        .select("key", "pub_type", F.explode("authors").alias("name"))
    )
    # A name with no identity is a node of its own
    node = F.concat(F.lit("n:"), F.col("name"))
    if alias_node_df is not None:
        papers = papers.join(
            alias_node_df, papers["name"] == alias_node_df["author_name"], "left"
        )
        node = F.coalesce(F.col("node_id"), node)
    return (
        papers.select("key", "pub_type", "name", node.alias("author"))
        .dropDuplicates(["key", "author"])
        .persist(StorageLevel.DISK_ONLY)
    )


# Self-join to find direct co-authors
def _direct_coauthor_pairs(paper_authors: DataFrame) -> DataFrame:
    left = paper_authors.alias("l")
    right = paper_authors.alias("r")
    return (
        left.join(
            right,
            (F.col("l.key") == F.col("r.key"))
            & (F.col("l.author") < F.col("r.author")),
        )
        .select(
            F.col("l.author").alias("a"),
            F.col("r.author").alias("b"),
            F.col("l.name").alias("name_a"),
            F.col("r.name").alias("name_b"),
            F.col("l.key").alias("evidence_paper"),
            F.col("l.pub_type").alias("pub_type"),
        )
        .persist(StorageLevel.DISK_ONLY)
    )


# One row per pair of people and orientation, since BFS walks edges either way
def _directed_edges(pairs: DataFrame) -> DataFrame:
    # Papers collapse to the first one, all the depth-2 evidence shows
    people = pairs.groupBy("a", "b").agg(
        F.min(F.struct("evidence_paper", "pub_type", "name_a", "name_b")).alias("rep")
    )
    return (
        people.select(
            F.col("a").alias("src"),
            F.col("b").alias("dst"),
            F.col("rep.name_a").alias("src_name"),
            F.col("rep.name_b").alias("dst_name"),
            F.col("rep.evidence_paper").alias("evidence_paper"),
            F.col("rep.pub_type").alias("pub_type"),
        )
        .union(
            people.select(
                F.col("b").alias("src"),
                F.col("a").alias("dst"),
                F.col("rep.name_b").alias("src_name"),
                F.col("rep.name_a").alias("dst_name"),
                F.col("rep.evidence_paper").alias("evidence_paper"),
                F.col("rep.pub_type").alias("pub_type"),
            )
        )
        .persist(StorageLevel.DISK_ONLY)
    )


# Every author's direct neighbors
def _seed_frontier(edges: DataFrame, source_authors: DataFrame | None) -> DataFrame:
    seed = edges.select(
        F.col("src").alias("source"),
        F.col("dst").alias("author"),
        F.lit(1).cast("int").alias("depth"),
        F.lit(1).cast("long").alias("n_paths"),
    )
    if source_authors is None:
        return seed
    return seed.join(
        F.broadcast(source_authors.withColumnRenamed("node_id", "source")),
        on="source",
        how="inner",
    )


# Drop authors from --max-hub-degree
def _edges_without_hub_pivots(
    edges: DataFrame, max_hub_degree: int | None
) -> DataFrame:
    if max_hub_degree is None:
        return edges

    hub_authors = (
        edges.groupBy("src")
        .agg(F.countDistinct("dst").alias("degree"))
        .filter(F.col("degree") > max_hub_degree)
        .select("src")
    )
    hub_count = hub_authors.count()
    logger.info(
        "Hub capping: %s author(s) exceed --max-hub-degree=%s and are excluded as hop pivots",
        f"{hub_count:,}",
        max_hub_degree,
    )
    if hub_count == 0:
        return edges

    capped = edges.join(F.broadcast(hub_authors), on="src", how="left_anti").persist(
        StorageLevel.DISK_ONLY
    )
    capped.count()
    return capped


# One hop of BFS over one slice of the frontier
def _expand_bucket(
    frontier: DataFrame, visited: DataFrame, edges: DataFrame, hop: int
) -> DataFrame:
    frontier = frontier.alias("f")
    edges = edges.select("src", "dst").alias("e")
    candidates = (
        frontier.join(edges, F.col("f.author") == F.col("e.src"))
        .select(
            F.col("f.source").alias("source"),
            F.col("e.dst").alias("author"),
            F.col("f.n_paths"),
        )
        .filter(F.col("source") != F.col("author"))
    )
    # Anti-join against everything already reached, then add up the chains that
    # reach each author at this level: each extends a shortest chain one hop shorter
    return (
        candidates.join(
            visited.select("source", "author").distinct(),
            on=["source", "author"],
            how="left_anti",
        )
        .groupBy("source", "author")
        .agg(F.sum("n_paths").alias("n_paths"))
        .select("source", "author", F.lit(hop).cast("int").alias("depth"), "n_paths")
        .persist(StorageLevel.DISK_ONLY)
    )


# Expand one BFS hop, or return None once nothing new is reachable
def _expand_hop(
    frontier: DataFrame,
    visited: DataFrame,
    edges: DataFrame,
    hop: int,
    source_buckets: int,
) -> DataFrame | None:
    bucket_of_source = F.pmod(
        F.crc32(F.col("source").cast("binary")), F.lit(source_buckets)
    )
    bucket_results: list[DataFrame] = []
    for bucket in range(source_buckets):
        frontier_slice = (
            frontier
            if source_buckets == 1
            else frontier.filter(bucket_of_source == bucket)
        )
        visited_slice = (
            visited
            if source_buckets == 1
            else visited.filter(bucket_of_source == bucket)
        )

        expanded = _expand_bucket(frontier_slice, visited_slice, edges, hop)
        expanded_count = expanded.count()
        if source_buckets > 1:
            logger.debug(
                "hop %s bucket %s/%s: %s new pairs",
                hop,
                bucket + 1,
                source_buckets,
                f"{expanded_count:,}",
            )
        if expanded_count == 0:
            expanded.unpersist()
            continue
        bucket_results.append(expanded)

    if not bucket_results:
        return None

    if len(bucket_results) == 1:
        return bucket_results[0]

    new_frontier = bucket_results[0]
    for part in bucket_results[1:]:
        new_frontier = new_frontier.unionByName(part)
    new_frontier = new_frontier.persist(StorageLevel.DISK_ONLY)
    new_frontier.count()  # Materialize before unpersisting buckets
    for part in bucket_results:
        part.unpersist()
    return new_frontier


def _keep_shortest(
    canonical: DataFrame, keys: list[str], aggregations: list[Column]
) -> DataFrame:
    min_depth = canonical.groupBy(*keys).agg(F.min("depth").alias("min_depth"))
    return (
        canonical.join(min_depth, keys)
        .filter(F.col("depth") == F.col("min_depth"))
        .groupBy(*keys)
        .agg(F.first("min_depth").alias("depth"), *aggregations)
    )


# If BFS reaches both (A, B) and (B, A), sort and fold together
def _merge_pairs(visited: DataFrame) -> DataFrame:
    canonical = visited.select(
        F.least("source", "author").alias("author_a"),
        F.greatest("source", "author").alias("author_b"),
        "depth",
        "n_paths",
    )

    # Both directions count the same chains, so take one
    return _keep_shortest(
        canonical, ["author_a", "author_b"], [F.max("n_paths").alias("n_paths")]
    ).persist(StorageLevel.DISK_ONLY)


# Evidence is rendered for depth 1 and 2 only, deeper pairs get null paths
def _with_evidence_paths(
    merged: DataFrame, pairs: DataFrame, edges: DataFrame
) -> DataFrame:
    # Depth 1: one path per shared paper, author_a < author_b matches `pairs`
    direct = (
        merged.filter(F.col("depth") == 1)
        .join(
            pairs,
            (F.col("author_a") == F.col("a")) & (F.col("author_b") == F.col("b")),
        )
        .groupBy("author_a", "author_b")
        .agg(
            F.collect_list(
                F.array(
                    hop_struct(
                        F.col("name_a"),
                        F.col("name_b"),
                        F.col("evidence_paper"),
                        F.col("pub_type"),
                    )
                )
            ).alias("paths")
        )
    )

    # Depth 2: the chain through the first common neighbor, one paper per hop
    first = edges.alias("e1")
    second = edges.alias("e2")
    via_one = (
        merged.filter(F.col("depth") == 2)
        .join(first, F.col("author_a") == F.col("e1.src"))
        .join(
            second,
            (F.col("e1.dst") == F.col("e2.src"))
            & (F.col("author_b") == F.col("e2.dst")),
        )
        .groupBy("author_a", "author_b")
        .agg(
            F.min(
                F.struct(
                    F.col("e1.dst").alias("via"),
                    hop_struct(
                        F.col("e1.src_name"),
                        F.col("e1.dst_name"),
                        F.col("e1.evidence_paper"),
                        F.col("e1.pub_type"),
                    ).alias("first"),
                    hop_struct(
                        F.col("e2.src_name"),
                        F.col("e2.dst_name"),
                        F.col("e2.evidence_paper"),
                        F.col("e2.pub_type"),
                    ).alias("second"),
                )
            ).alias("chain")
        )
        .select(
            "author_a",
            "author_b",
            F.array(F.array(F.col("chain.first"), F.col("chain.second"))).alias(
                "paths"
            ),
        )
    )

    return merged.join(
        direct.unionByName(via_one), ["author_a", "author_b"], "left"
    ).persist(StorageLevel.DISK_ONLY)


# Weight of the shortest depth times the number of shortest paths
def _score_pairs(merged: DataFrame, weights: dict[str, int]) -> DataFrame:
    score_by_depth = lit_map(
        {weight_depth(key): weight for key, weight in weights.items()}
    )
    type_by_depth = lit_map(COI_TYPE_MAP)
    # Direct pairs count their papers, deeper ones count chains of people
    return (
        merged.withColumn(
            "evidence_path_count",
            F.when(F.col("depth") == 1, F.size("paths")).otherwise(F.col("n_paths")),
        )
        .withColumn(
            "coi_score",
            score_by_depth[F.col("depth")] * F.col("evidence_path_count"),
        )
        .withColumn("coi_type", type_by_depth[F.col("depth")])
    )


# Phases: bipartite graph, direct co-author self-join, BFS for submarine COIs, merge
def _analyze(
    spark: SparkSession,
    parquet_path: str,
    weights: dict[str, int],
    checkpoint_dir: str,
    max_depth: int,
    source_buckets: int,
    max_hub_degree: int | None,
    source_authors: DataFrame | None,
    alias_node_df: DataFrame | None,
) -> DataFrame:
    logger.info(
        "Phase 1/4: reading papers (bipartite paper<->author edges), max_depth=%s, source_buckets=%s",
        max_depth,
        source_buckets,
    )
    paper_authors = _read_paper_author_edges(spark, parquet_path, alias_node_df)
    logger.info("Phase 1/4 done: %s (paper, author) rows", f"{paper_authors.count():,}")

    logger.info("Phase 2/4: self-join for depth-1 (direct co-author) pairs ...")
    pairs = _direct_coauthor_pairs(paper_authors)
    logger.info(
        "Phase 2/4 done: %s direct co-author pairs (depth 1)", f"{pairs.count():,}"
    )
    paper_authors.unpersist()

    edges = _directed_edges(pairs)
    edges.count()
    expansion_edges = _edges_without_hub_pivots(edges, max_hub_degree)

    frontier = _materialize(
        spark,
        _seed_frontier(edges, source_authors),
        _checkpoint_path(checkpoint_dir, "frontier_seed"),
    )
    visited = frontier
    previous_frontier: DataFrame | None = None
    # Alternate the visited table between two paths so at most two copies sit on disk
    visited_paths = [
        _checkpoint_path(checkpoint_dir, "visited_a"),
        _checkpoint_path(checkpoint_dir, "visited_b"),
    ]

    for hop in range(2, max_depth + 1):
        logger.info("Phase 3/4: hop %s/%s ...", hop, max_depth)
        new_frontier = _expand_hop(
            frontier, visited, expansion_edges, hop, source_buckets
        )
        if new_frontier is None:
            logger.info(
                "Phase 3/4: hop %s/%s done: 0 new pairs, BFS exhausted - stopping early",
                hop,
                max_depth,
            )
            break
        logger.info(
            "Phase 3/4: hop %s/%s done: %s new (source, author) pairs",
            hop,
            max_depth,
            f"{new_frontier.count():,}",
        )

        if previous_frontier is not None:
            previous_frontier.unpersist()
        previous_frontier = frontier
        visited = _materialize(
            spark, visited.unionByName(new_frontier), visited_paths[hop % 2]
        )
        frontier = new_frontier

    if previous_frontier is not None:
        previous_frontier.unpersist()
    if expansion_edges is not edges:
        expansion_edges.unpersist()

    logger.info(
        "Phase 4/4: merging (canonicalize pairs, dedupe, keep shortest depth) ..."
    )
    merged = _merge_pairs(visited)
    with_evidence = _with_evidence_paths(merged, pairs, edges)
    with_evidence.count()  # Materialize before unpersisting what the evidence reads
    merged.unpersist()
    edges.unpersist()
    pairs.unpersist()
    logger.info(
        "Phase 4/4 done: merging complete (pair count logged after email resolution)"
    )

    return _score_pairs(with_evidence, weights).select(
        "author_a",
        "author_b",
        "depth",
        "coi_type",
        "coi_score",
        "evidence_path_count",
        "paths",
    )


@contextmanager
def coi_analysis(
    spark: SparkSession,
    parquet_path: str,
    weights: dict[str, int],
    max_depth: int = 3,
    checkpoint_dir: str | None = None,
    source_buckets: int = 1,
    max_hub_degree: int | None = None,
    source_authors: DataFrame | None = None,
    alias_node_df: DataFrame | None = None,
) -> Iterator[DataFrame]:
    """
    `source_buckets` splits each BFS hop into that many sequential passes
    over disjoint slices.

    `alias_node_df` (author_name, node_id) folds the aliases of one person into
    one node. A name without a row stays its own node, and `source_authors`
    holds the node ids (column `node_id`) to search from.
    """
    if source_buckets < 1:
        raise ValueError(f"source_buckets must be at least 1, got {source_buckets}")

    own_checkpoint_dir = checkpoint_dir is None
    checkpoint_dir = checkpoint_dir or tempfile.mkdtemp(prefix="coi_ckpt_")
    if "://" not in checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    try:
        yield _analyze(
            spark,
            parquet_path,
            weights,
            checkpoint_dir=checkpoint_dir,
            max_depth=max_depth,
            source_buckets=source_buckets,
            max_hub_degree=max_hub_degree,
            source_authors=source_authors,
            alias_node_df=alias_node_df,
        )
    finally:
        if own_checkpoint_dir:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


# `mapping` is keyed on the node id, one row per known person. Every node reported
# here is "u:" + email, so author_a < author_b also orders email_a < email_b
def resolve_emails(df_coi: DataFrame, mapping: DataFrame) -> DataFrame:
    known_a = F.broadcast(mapping.alias("m1"))
    known_b = F.broadcast(mapping.alias("m2"))
    return (
        df_coi.join(known_a, df_coi["author_a"] == F.col("m1.node_id"))
        .join(known_b, df_coi["author_b"] == F.col("m2.node_id"))
        .select(
            F.col("m1.author_name").alias("author_a"),
            F.col("m1.email").alias("email_a"),
            F.col("m1.dblp_id").alias("dblp_id_a"),
            F.col("m1.disam_level").alias("disam_level_a"),
            F.col("m2.author_name").alias("author_b"),
            F.col("m2.email").alias("email_b"),
            F.col("m2.dblp_id").alias("dblp_id_b"),
            F.col("m2.disam_level").alias("disam_level_b"),
            "depth",
            "coi_type",
            "coi_score",
            "evidence_path_count",
            "paths",
        )
    )


# A pair is reported once however many aliases its two people have
def check_unique_pairs(df_resolved: DataFrame) -> None:
    duplicated = (
        df_resolved.groupBy(
            F.least("email_a", "email_b").alias("a"),
            F.greatest("email_a", "email_b").alias("b"),
        )
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicated:
        logger.error(
            "%s email pair(s) appear in more than one row, a person is split over several nodes",
            f"{duplicated:,}",
        )
        raise RuntimeError(f"{duplicated:,} email pair(s) are duplicated")
