"""Pipeline configuration and stages, independent of invokation.

`main.py` parses the command line and dispatches, everything about what the run
does lives here, reached through the dataclasses below instead of an argparse
Namespace. That keeps the stages testable and every derived path in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import cache, webhdfs
from .coi import check_unique_pairs, coi_analysis, resolve_emails
from .constants import COI_TYPE_MAP, DEFAULT_WEIGHTS, weight_depth
from .dblp import (
    cache_params,
    dump_cache_path,
    homepages_path,
    load_homepages,
    stream_dblp,
)
from .evidence import format_paths
from .mappings import (
    HdfsUpload,
    build_alias_node_df,
    build_alias_pid_df,
    build_email_mapping_df,
    build_node_mapping_df,
)
from .outputs import (
    HdfsTmp,
    write_analysis_csv,
    write_coi_csv,
    write_unresolved_users_csv,
)
from .publications import is_combined_cache_stale, merge_and_dedup_parquets
from .qualify import check_reviewer_qualifications
from .semantic_scholar_authors import (
    build_dblp_name_map,
    build_enriched_homepages,
    load_semantic_scholar_authors,
    stream_semantic_scholar_authors,
)
from .semantic_scholar_papers import stream_semantic_scholar_papers
from .users import (
    ResolvedUser,
    build_author_email_index,
    resolve_users,
    unresolved_users,
)

logger = logging.getLogger(__name__)


# The cache path derived from a dump's path
def _default_cache_path(source: str, suffix: str) -> str:
    return str(Path(source.rstrip("/")).with_suffix(suffix))


@dataclass(frozen=True)
class SourceConfig:
    """Every input dump, the cache each lands in, and the cutoff they share.

    Shared by the `coi` and `qualify` subcommands.
    """

    dblp_xml: str
    users_csv: str
    cutoff_year: int
    force: bool = False
    dblp_parquet_override: str | None = None
    s2_authors_dir: str | None = None
    s2_authors_parquet_override: str | None = None
    s2_papers_dir: str | None = None
    s2_papers_parquet_override: str | None = None
    combined_parquet_override: str | None = None

    @classmethod
    def from_args(cls, args, cutoff_year: int) -> SourceConfig:
        return cls(
            dblp_xml=args.dblp_xml,
            users_csv=args.users,
            cutoff_year=cutoff_year,
            force=args.force,
            dblp_parquet_override=args.parquet,
            s2_authors_dir=args.s2_authors_dir,
            s2_authors_parquet_override=args.s2_authors_parquet,
            s2_papers_dir=args.s2_papers_dir,
            s2_papers_parquet_override=args.s2_papers_parquet,
            combined_parquet_override=args.combined_parquet,
        )

    @property
    def dblp_parquet(self) -> str:
        return self.dblp_parquet_override or dump_cache_path(self.dblp_xml)

    @property
    def s2_authors_parquet(self) -> str | None:
        if not self.s2_authors_dir:
            return None
        return self.s2_authors_parquet_override or _default_cache_path(
            self.s2_authors_dir, ".parquet"
        )

    @property
    def s2_papers_parquet(self) -> str | None:
        if not self.s2_papers_dir:
            return None
        return self.s2_papers_parquet_override or _default_cache_path(
            self.s2_papers_dir, ".parquet"
        )

    @property
    def combined_parquet(self) -> str:
        return self.combined_parquet_override or _default_cache_path(
            self.dblp_parquet, ".combined.parquet"
        )

    @property
    def cache_params(self) -> dict[str, int]:
        return cache_params(self.cutoff_year)


@dataclass(frozen=True)
class ClusterConfig:
    """Where the Spark job runs and how driver-built data reaches it."""

    hdfs_parquet: str | None = None
    webhdfs: str | None = None

    @classmethod
    def from_args(cls, args) -> ClusterConfig:
        return cls(hdfs_parquet=args.hdfs_parquet, webhdfs=args.webhdfs)

    @property
    def enabled(self) -> bool:
        return self.hdfs_parquet is not None

    @property
    def base(self) -> str | None:
        """The HDFS directory holding this run's staged files, or None locally."""
        if not self.hdfs_parquet:
            return None
        return self.hdfs_parquet.rsplit("/", 1)[0]

    def upload_target(self, local_parquet: str, name: str) -> HdfsUpload | None:
        """Where a driver-built table should be staged, or None in local mode.

        The staging file sits beside the cache it was derived from.
        """
        if self.base is None or self.webhdfs is None:
            return None
        return (
            f"{local_parquet}.{name}.parquet",
            f"{self.base}/{name}.parquet",
            self.webhdfs,
        )

    def csv_target(self, name: str) -> HdfsTmp | None:
        """Where Spark should write a CSV before it is pulled back."""
        if self.base is None or self.webhdfs is None:
            return None
        return (f"{self.base}/csv_out/{name}", self.webhdfs)


@dataclass(frozen=True)
class OutputConfig:
    """Where results go, and which of them are produced at all."""

    output: str
    unresolved_output: str
    threshold: int
    analysis: bool = False
    analysis_output: str = "coi-analysis.csv"

    @classmethod
    def from_args(cls, args, threshold: int) -> OutputConfig:
        return cls(
            output=args.output,
            unresolved_output=args.unresolved_output,
            threshold=threshold,
            analysis=args.analysis,
            analysis_output=args.analysis_output,
        )


@dataclass(frozen=True)
class Artefacts:
    """Parquet paths."""

    publications_parquet: str
    homepages_parquet: str


def check_depths_are_scored(weights: dict[str, int], max_depth: int) -> None:
    """Refuse a --max-depth without a weight and COI type for every depth.

    A depth with neither would produce a null coi_score.
    """
    depths = range(1, max_depth + 1)
    missing_weights = [
        depth for depth in depths if depth not in {weight_depth(k) for k in weights}
    ]
    if missing_weights:
        listed = ", ".join(str(depth) for depth in missing_weights)
        example = ", ".join(f"COI_WEIGHT_DEPTH_{depth}" for depth in missing_weights)
        raise SystemExit(
            f"--max-depth {max_depth} has no weight for depth(s) {listed}; "
            f"set {example} (see .env.example)"
        )
    missing_types = [depth for depth in depths if depth not in COI_TYPE_MAP]
    if missing_types:
        listed = ", ".join(str(depth) for depth in missing_types)
        raise SystemExit(
            f"no COI type is defined for depth(s) {listed}; "
            f"extend COI_TYPE_MAP in constants.py before raising --max-depth"
        )


def build_artefacts(config: SourceConfig) -> Artefacts:
    """Parse every configured dump into its cache, return what to analyse.

    Enriches the homepage index with S2 identity data whenever --s2-authors-dir
    is given, with or without --s2-papers-dir.

    `source_parquets` collects every cache that carries graph edges, DBLP first
    so it wins DOI collisions. They are only combined when there is more than one.
    """
    stream_dblp(
        config.dblp_xml,
        config.dblp_parquet,
        min_year=config.cutoff_year,
        force=config.force,
    )
    homepages_parquet = homepages_path(config.dblp_parquet)

    authors_dir = config.s2_authors_dir
    authors_parquet = config.s2_authors_parquet
    if authors_dir and authors_parquet:
        stream_semantic_scholar_authors(
            authors_dir, authors_parquet, force=config.force
        )
        homepages_parquet = build_enriched_homepages(
            homepages_parquet, authors_parquet, force=config.force
        )

    source_parquets = [config.dblp_parquet]
    papers_dir = config.s2_papers_dir
    papers_parquet = config.s2_papers_parquet
    if papers_dir and papers_parquet:
        stream_semantic_scholar_papers(
            papers_dir,
            papers_parquet,
            lambda: _build_name_map(authors_parquet),
            min_year=config.cutoff_year,
            force=config.force,
            name_map_source=authors_parquet,
        )
        source_parquets.append(papers_parquet)

    if len(source_parquets) == 1:
        return Artefacts(config.dblp_parquet, homepages_parquet)

    combined = config.combined_parquet
    check_combined_cache(combined, config)
    if config.force or is_combined_cache_stale(combined, source_parquets):
        merge_and_dedup_parquets(source_parquets, combined, config.cache_params)
    else:
        logger.info("Combined publications Parquet up-to-date, skipping merge")
    return Artefacts(combined, homepages_parquet)


def _build_name_map(authors_parquet: str | None) -> dict[str, list[str]]:
    if not authors_parquet:
        return {}
    name_map = build_dblp_name_map(load_semantic_scholar_authors(authors_parquet))
    logger.info(
        "%s S2 author(s) carry an authoritative DBLP name mapping",
        f"{len(name_map):,}",
    )
    return name_map


def check_combined_cache(combined: str, config: SourceConfig) -> None:
    cache.check_parameters(
        combined,
        config.cache_params,
        force=config.force,
        description="The combined publications cache",
    )


def build_spark_session(args, spark_conf: dict[str, str]) -> SparkSession:
    builder = (
        SparkSession.builder.appName("COI-System")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    )
    for key, value in spark_conf.items():
        builder = builder.config(key, value)

    if args.spark_local_dir:
        builder = builder.config("spark.local.dir", args.spark_local_dir)
    if args.executor_memory:
        builder = builder.config("spark.executor.memory", args.executor_memory)
    if args.executor_cores:
        builder = builder.config("spark.executor.cores", str(args.executor_cores))
    # Only pin local cores on request, so spark-submit's --master isn't overridden
    if args.local_cores is not None:
        builder = builder.master(f"local[{args.local_cores}]")
    if args.hdfs:
        builder = builder.config("spark.hadoop.fs.defaultFS", args.hdfs)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stage_parquet_on_hdfs(
    local_parquet: str, hdfs_parquet: str, webhdfs_base: str
) -> None:
    """Push the local Parquet cache to HDFS so the executors can read it.

    The parse writes via plain pyarrow, which has no HDFS support.
    """
    logger.info("Uploading local parquet cache -> %s via WebHDFS ...", hdfs_parquet)
    webhdfs.upload_file(local_parquet, webhdfs.uri_to_path(hdfs_parquet), webhdfs_base)


def log_pair_statistics(df_coi: DataFrame) -> None:
    logger.info(
        "%s COI pairs found (any endpoint, before email resolution)",
        f"{df_coi.count():,}",
    )
    # An agg without groupBy yields one row, the guard is for type checker
    max_paths_row = df_coi.agg(F.max("evidence_path_count").alias("max_paths")).first()
    max_paths = max_paths_row["max_paths"] if max_paths_row else None
    logger.info(
        "Most shortest paths counted for one pair: %s "
        "(papers for depth-1 pairs, chains of people for deeper ones)",
        max_paths,
    )


def resolve_and_format(
    df_coi: DataFrame,
    mapping: DataFrame,
    alias_pid_df: DataFrame,
) -> tuple[DataFrame, list[DataFrame]]:
    """Keep the known pairs, resolve them to email addresses, render evidence.

    Returns the DataFrame and every table still held in cache for it,
    for the caller to release once the outputs are written.
    """
    # Persisted to avoid OOM
    df_coi = df_coi.persist(StorageLevel.DISK_ONLY)
    log_pair_statistics(df_coi)

    df_pairs = resolve_emails(df_coi, mapping).persist(StorageLevel.DISK_ONLY)
    logger.info("%s pairs resolved to known email addresses", f"{df_pairs.count():,}")
    check_unique_pairs(df_pairs)

    logger.info("Formatting evidence paths for resolvable pairs ...")
    df_formatted = format_paths(df_pairs, alias_pid_df).persist(StorageLevel.DISK_ONLY)
    logger.info(
        "%s email-resolvable COI relationships found", f"{df_formatted.count():,}"
    )
    return df_formatted, [df_coi, df_pairs, df_formatted]


def write_outputs(
    df_resolved: DataFrame, outputs: OutputConfig, cluster: ClusterConfig
) -> None:
    write_coi_csv(
        df_resolved,
        outputs.output,
        outputs.threshold,
        hdfs_tmp=cluster.csv_target("coi"),
    )
    logger.info("Written -> %s (threshold=%s)", outputs.output, outputs.threshold)

    if outputs.analysis:
        write_analysis_csv(
            df_resolved,
            outputs.analysis_output,
            hdfs_tmp=cluster.csv_target("coi-analysis"),
        )
        logger.info("Written -> %s", outputs.analysis_output)


def report_unresolved_users(
    resolved: list[ResolvedUser], outputs: OutputConfig
) -> None:
    """Write out the users whose identity is unclear."""
    unresolved = unresolved_users(resolved)
    write_unresolved_users_csv(unresolved, outputs.unresolved_output)
    logger.info(
        "%s of %s user(s) could not be disambiguated and are excluded from the "
        "analysis; listed in %s",
        f"{len(unresolved):,}",
        f"{len(resolved):,}",
        outputs.unresolved_output,
    )


def run_coi(
    args,
    sources: SourceConfig,
    cluster: ClusterConfig,
    outputs: OutputConfig,
    spark_conf: dict[str, str],
) -> None:
    """The full COI run."""
    weights = DEFAULT_WEIGHTS
    check_depths_are_scored(weights, args.max_depth)

    artefacts = build_artefacts(sources)
    spark = build_spark_session(args, spark_conf)

    analysis_parquet = artefacts.publications_parquet
    hdfs_parquet, webhdfs_base = cluster.hdfs_parquet, cluster.webhdfs
    if hdfs_parquet and webhdfs_base:
        stage_parquet_on_hdfs(
            artefacts.publications_parquet, hdfs_parquet, webhdfs_base
        )
        analysis_parquet = hdfs_parquet

    alias_pid_df = build_alias_pid_df(
        spark,
        artefacts.homepages_parquet,
        hdfs_upload=cluster.upload_target(sources.dblp_parquet, "alias_pid"),
    )

    logger.info("Resolving DBLP authors to email addresses via Users.csv ...")
    resolved = resolve_users(
        load_homepages(artefacts.homepages_parquet), sources.users_csv
    )
    email_index = build_author_email_index(resolved)
    report_unresolved_users(resolved, outputs)
    email_mapping = build_email_mapping_df(
        spark,
        email_index,
        hdfs_upload=cluster.upload_target(sources.dblp_parquet, "email_mapping"),
    )
    alias_node_df = build_alias_node_df(alias_pid_df, email_mapping)
    mapping = build_node_mapping_df(email_mapping).persist(StorageLevel.DISK_ONLY)

    logger.info(
        "Running distributed BFS scoped to %s known user(s) (max depth %s) ...",
        f"{len({email for email, _, _ in email_index.values()}):,}",
        args.max_depth,
    )
    try:
        with coi_analysis(
            spark,
            analysis_parquet,
            weights,
            max_depth=args.max_depth,
            checkpoint_dir=args.checkpoint_dir,
            source_buckets=args.source_buckets,
            max_hub_degree=args.max_hub_degree,
            source_authors=mapping.select("node_id"),
            alias_node_df=alias_node_df,
        ) as df_coi:
            df_resolved, cached = resolve_and_format(df_coi, mapping, alias_pid_df)
            try:
                write_outputs(df_resolved, outputs, cluster)
            finally:
                for df in cached:
                    df.unpersist()
    finally:
        mapping.unpersist()
    logger.info("Done.")


def run_qualify(
    sources: SourceConfig, output: str, reverse: bool, unresolved_output: str
) -> None:
    """The reviewer qualification run."""
    artefacts = build_artefacts(sources)
    check_reviewer_qualifications(
        artefacts.publications_parquet,
        sources.users_csv,
        output,
        reverse=reverse,
        homepages_parquet=artefacts.homepages_parquet,
        unresolved_output_path=unresolved_output,
    )
