from __future__ import annotations

import pytest

from coi_dams.constants import DEFAULT_WEIGHTS
from coi_dams.flags import build_coi_parser, validate_coi_args
from coi_dams.pipeline import (
    ClusterConfig,
    OutputConfig,
    SourceConfig,
    check_depths_are_scored,
)

BASE_COI_ARGS = ["--dblp-xml", "dblp.xml", "--users", "Users.csv"]


def _sources(**overrides) -> SourceConfig:
    defaults = {
        "dblp_xml": "/data/dblp.xml",
        "users_csv": "/input/Users.csv",
        "cutoff_year": 2016,
    }
    return SourceConfig(**{**defaults, **overrides})


def test_dblp_parquet_default():
    assert _sources().dblp_parquet == "/data/dblp.parquet"


def test_combined_parquet_default():
    assert _sources().combined_parquet == "/data/dblp.combined.parquet"


def test_source_config_from_args():
    args = build_coi_parser().parse_args(
        [*BASE_COI_ARGS, "--s2-authors-dir", "/data/authors", "--force"]
    )
    sources = SourceConfig.from_args(args, cutoff_year=2016)
    assert sources.dblp_xml == "dblp.xml"
    assert sources.users_csv == "Users.csv"
    assert sources.force is True
    assert sources.s2_authors_parquet == "/data/authors.parquet"


def test_upload_target():
    cluster = ClusterConfig(
        hdfs_parquet="hdfs://nn:9000/coi/dblp.parquet", webhdfs="http://nn:9870"
    )
    local, remote, base = cluster.upload_target("/data/dblp.parquet", "alias_pid")
    # The staging file sits next to the cache it was derived from
    assert local == "/data/dblp.parquet.alias_pid.parquet"
    assert remote == "hdfs://nn:9000/coi/alias_pid.parquet"
    assert base == "http://nn:9870"


def test_unscored_depth():
    """A depth without a weight is rejected instead of producing a null score."""
    with pytest.raises(SystemExit) as error:
        check_depths_are_scored({"depth_1": 100}, max_depth=3)
    message = str(error.value)
    assert "depth(s) 2, 3" in message
    assert "COI_WEIGHT_DEPTH_2" in message


def test_depth_without_coi_type():
    weights = {**DEFAULT_WEIGHTS, "depth_5": 1, "depth_6": 1}
    with pytest.raises(SystemExit) as error:
        check_depths_are_scored(weights, max_depth=6)
    assert "COI_TYPE_MAP" in str(error.value)


@pytest.mark.parametrize(
    "flag",
    ["--source-buckets", "--max-depth", "--shuffle-partitions", "--max-hub-degree"],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_counts_below_one_are_rejected(flag, value):
    """--source-buckets 0 is rejected."""
    with pytest.raises(SystemExit):
        build_coi_parser().parse_args([*BASE_COI_ARGS, flag, value])


def test_max_depth_limit():
    with pytest.raises(SystemExit):
        build_coi_parser().parse_args([*BASE_COI_ARGS, "--max-depth", "5"])


def test_xml_is_required():
    with pytest.raises(SystemExit):
        build_coi_parser().parse_args(["--users", "Users.csv"])


def test_threshold_from_env():
    args = build_coi_parser().parse_args(BASE_COI_ARGS)
    outputs = OutputConfig.from_args(args, threshold=args.threshold)
    assert outputs.threshold == args.threshold


def test_hdfs_parquet_needs_webhdfs():
    args = build_coi_parser().parse_args(
        [*BASE_COI_ARGS, "--hdfs-parquet", "hdfs://nn:9000/coi/dblp.parquet"]
    )
    with pytest.raises(SystemExit, match="--webhdfs"):
        validate_coi_args(args)
