from __future__ import annotations

import argparse

from .constants import COI_ACCEPT_THRESHOLD, REVIEWER_MIN_PAPERS
from .qualify import QUALIFYING_VENUES


# Rejects integers below 1
def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {value}")
    return value


# Limit --max-depth at MAX_DEPTH_LIMIT
def _max_depth(raw: str) -> int:
    MAX_DEPTH_LIMIT = 4  # Arbitrarily chosen
    value = _positive_int(raw)
    if value > MAX_DEPTH_LIMIT:
        raise argparse.ArgumentTypeError(
            f"must be {MAX_DEPTH_LIMIT} or less, got {value}. Higher values don't deliver useful data, and take significantly longer to compute"
        )
    return value


# Flags shared by coi and qualify
def _add_dblp_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dblp-xml",
        required=True,
        metavar="XML_PATH",
        help="Path to the extracted dblp.xml. The definitions file dblp.dtd must be in the same "
        "directory. Download both using misc/fetch_data_sources.py",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        metavar="PARQUET_PATH",
        help="Parquet cache path (default: --dblp-xml with a .parquet suffix). XML parsing doesn't use Spark so this needs to be a local path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Parquet caches even if they're up-to-date",
    )


# Flags for Semantic Scholar
def _add_semantic_scholar_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--s2-authors-dir",
        default=None,
        metavar="DIR",
        help="Directory of Semantic Scholar 'authors' dataset (part-*.jsonl.gz files). "
        "Enriches DBLP disambiguation with S2 affiliations",
    )
    parser.add_argument(
        "--s2-papers-dir",
        default=None,
        metavar="DIR",
        help="Directory of Semantic Scholar 'papers' dataset (part-*.jsonl.gz files). "
        "Adds S2 co-authorship edges to the COI graph, merged with DBLP's",
    )
    parser.add_argument(
        "--s2-authors-parquet",
        default=None,
        metavar="PARQUET_PATH",
        help="Local Parquet cache for the parsed S2 authors dump "
        "(default: --s2-authors-dir with a .parquet suffix)",
    )
    parser.add_argument(
        "--s2-papers-parquet",
        default=None,
        metavar="PARQUET_PATH",
        help="Local Parquet cache for the parsed S2 papers dump "
        "(default: --s2-papers-dir with a .parquet suffix)",
    )
    parser.add_argument(
        "--combined-parquet",
        default=None,
        metavar="PARQUET_PATH",
        help="Path to write the merged DBLP and S2 publications Parquet. (default: --parquet with a "
        ".combined.parquet suffix). Used if --s2-papers-dir is set",
    )


def _add_coi_args(parser: argparse.ArgumentParser) -> None:
    _add_dblp_source_args(parser)
    _add_semantic_scholar_args(parser)
    parser.add_argument(
        "--users",
        required=True,
        metavar="CSV_PATH",
        help="Path to the users CSV, on which the computation is done",
    )
    parser.add_argument(
        "--output",
        default="coi.csv",
        metavar="CSV_PATH",
        help="Tab-separated email pairs of conflicts above a certain threshold",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Also write a detailed CSV (all scores, evidence, disambiguation certainty) for analysing the output",
    )
    parser.add_argument(
        "--analysis-output",
        default="coi-analysis.csv",
        metavar="CSV_PATH",
        help="Path where --analysis should write the detailed report (default: coi-analysis.csv)",
    )
    parser.add_argument(
        "--unresolved-output",
        default="unresolved-users.csv",
        metavar="CSV_PATH",
        help="CSV of users whose DBLP identity could not be disambiguated (default: unresolved-users.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=COI_ACCEPT_THRESHOLD,
        metavar="SCORE",
        help="COI pairs scoring below this are omitted from --output and thus "
        f"treated as non-conflicts (default: {COI_ACCEPT_THRESHOLD}, from "
        "COI_ACCEPT_THRESHOLD; see .env.example). The comparison is inclusive",
    )
    parser.add_argument(
        "--max-depth",
        type=_max_depth,
        default=2,
        help="Longest co-authorship chain to follow (default: 2, max: 4). Weights come from COI_WEIGHT_DEPTH_N env vars (see .env.example)",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Spark checkpoint dir")

    parser.add_argument(
        "--driver-memory",
        default="4g",
        help="Spark driver JVM heap in local mode (default: 4g)",
    )
    parser.add_argument(
        "--executor-memory",
        default=None,
        metavar="SIZE",
        help="spark.executor.memory on a real cluster (default: unset, so Spark's 1g default, which is too small for most cases. Check the workers' free memory in the master UI and set accordingly)",
    )
    parser.add_argument(
        "--executor-cores",
        type=_positive_int,
        default=None,
        help="spark.executor.cores - fewer cores per executor means fewer concurrency in the same heap, lowering memory pressure per executor. Lower this if executors are OOMing despite enough --executor-memory",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=_positive_int,
        default=200,
        help="spark.sql.shuffle.partitions - raise for large joins so each task's spill buffer stays small (default: 200)",
    )
    parser.add_argument(
        "--local-cores",
        type=_positive_int,
        default=None,
        help="How many local cores to use (default: unset, so Spark default/cluster master). Lower this to reduce memory usage on a memory-constrained machine",
    )
    parser.add_argument(
        "--spark-local-dir",
        default=None,
        metavar="PATH",
        help="spark.local.dir - where each executor spills shuffle/persists data. Point this at the largest available mount/filesystem. (default: /tmp, but that can fill RAM quickly if you're using tmpfs, or the disk, if not)",
    )

    parser.add_argument(
        "--hdfs",
        default=None,
        metavar="URI",
        help="spark.hadoop.fs.defaultFS - HDFS URI, e.g. hdfs://10.0.0.0:9000. Needed so relative/hdfs:// paths resolve on the cluster",
    )
    parser.add_argument(
        "--hdfs-parquet",
        default=None,
        metavar="URI",
        help="hdfs:// path to sync the local --parquet cache before analysis, and read from for the actual Spark job (e.g. hdfs://10.0.0.0:9000/coi/dblp.parquet). Needed for a cluster run since workers can't access the laptop's local.disk",
    )
    parser.add_argument(
        "--webhdfs",
        default=None,
        metavar="URL",
        help="WebHDFS base URL used to upload the local parquet cache to --hdfs-parquet (e.g. http://10.0.0.0:9870). Required if --hdfs-parquet is set",
    )

    parser.add_argument(
        "--source-buckets",
        type=_positive_int,
        default=1,
        help="Split each BFS hop into this many sequential bucket passes, so at most 1/N of the graph's frontier is in memory/shuffle at once. Raise this to increase computation time but lower memory pressure (default: 1, no bucketing)",
    )
    parser.add_argument(
        "--max-hub-degree",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Exclude authors with more than N distinct co-authors from acting as a 'universal connector' during submarine-COI calculation. A few very high degree hub authors can increase computation time, and shuffle/disk usage. This is a 'lossy' parameter: some submarine COIs are missed. (default: unset)",
    )


def _add_qualify_args(parser: argparse.ArgumentParser) -> None:
    _add_dblp_source_args(parser)
    _add_semantic_scholar_args(parser)
    parser.add_argument(
        "--users",
        required=True,
        metavar="CSV_PATH",
        help="Path to the input CSV with users",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="CSV_PATH",
        help="Output CSV (default: qualified-users.csv, or unqualified-users.csv with --reverse)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Output a list of unqualified reviewers instead of qualified",
    )
    parser.add_argument(
        "--unresolved-output",
        default="unresolved-users.csv",
        metavar="CSV_PATH",
        help="CSV of users whose DBLP identity could not be disambiguated; "
        "excluded from --output either way (default: unresolved-users.csv)",
    )


def validate_coi_args(args: argparse.Namespace) -> None:
    if args.hdfs_parquet and not args.webhdfs:
        raise SystemExit(
            "--hdfs-parquet requires --webhdfs (e.g. http://10.0.0.0:9870)"
        )


def build_coi_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coi",
        description="COI detection for all authors via constructing a graph from DBLP and running distributed BFS",
    )
    _add_coi_args(parser)
    return parser


def build_qualify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qualify",
        # Built from the configuration so the help text can't drift
        description=(
            "Check if the reviewers have published at least "
            f"{REVIEWER_MIN_PAPERS} paper(s) in: "
            f"{'/'.join(QUALIFYING_VENUES)} "
            "(set by COI_REVIEWER_MIN_PAPERS and COI_QUALIFYING_VENUES, "
            "see .env.example)"
        ),
    )
    _add_qualify_args(parser)
    return parser
