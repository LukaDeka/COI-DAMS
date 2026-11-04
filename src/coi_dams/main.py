"""Console entry points: parse the command line, hand off to the pipeline.

This module only knows how a run is invoked, what it does lives in pipeline.py.
"""

from __future__ import annotations

import logging
import sys

from .constants import OLDEST_CUTOFF_YEAR
from .flags import build_coi_parser, build_qualify_parser, validate_coi_args
from .pipeline import ClusterConfig, OutputConfig, SourceConfig, run_coi, run_qualify

logger = logging.getLogger(__name__)

# Spark settings that belong to this workload, per-run settings are CLI flags
SPARK_CONF = {
    "spark.memory.fraction": "0.7",
    "spark.memory.storageFraction": "0.4",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    # Headroom for the driver's task metadata, the driver is a 15GB laptop
    "spark.driver.maxResultSize": "3g",
    # Old checkpoint and shuffle blocks pile up on disk otherwise
    "spark.cleaner.referenceTracking.cleanCheckpoints": "true",
    "spark.worker.cleanup.enabled": "true",
    # Pinned so a firewall can allow-list exactly these two ports
    "spark.driver.port": "7078",
    "spark.driver.blockManager.port": "7079",
    # Both workers share a host IP, so Spark would read shuffle blocks from a peer's /tmp
    "spark.shuffle.readHostLocalDisk": "false",
    # The hop join is skewed by hub authors, AQE splits the oversized partitions
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    # With 820M+ rows the 64m default gave 1000+ tasks per stage, and their
    # MapStatus metadata surpassed 12GB and crashed the driver
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64m",
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main_coi() -> None:
    """Entry point for the `coi` feature."""
    _configure_logging()
    args = build_coi_parser().parse_args()
    validate_coi_args(args)
    run_coi(
        args,
        sources=SourceConfig.from_args(args, cutoff_year=OLDEST_CUTOFF_YEAR),
        cluster=ClusterConfig.from_args(args),
        outputs=OutputConfig.from_args(args, threshold=args.threshold),
        spark_conf=SPARK_CONF,
    )


def main_qualify() -> None:
    """Entry point for the `qualify` console script."""
    _configure_logging()
    args = build_qualify_parser().parse_args()
    output = args.output or (
        "unqualified-users.csv" if args.reverse else "qualified-users.csv"
    )
    run_qualify(
        SourceConfig.from_args(args, cutoff_year=OLDEST_CUTOFF_YEAR),
        output,
        reverse=args.reverse,
        unresolved_output=args.unresolved_output,
    )


# TODO: rewrite this with a main "main" and based on the arg run coi or qualify feature
def main():
    _configure_logging()
    args = build_qualify_parser().parse_args()


if __name__ == "__main__":
    # TODO: don't explicitly only call main_coi()
    main_coi()
