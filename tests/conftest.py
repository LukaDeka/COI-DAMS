from __future__ import annotations

import shutil

import pytest

# Spark-backed tests are skipped where there is no JVM
requires_java = pytest.mark.skipif(
    shutil.which("java") is None, reason="Spark needs a JVM on PATH"
)


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.appName("coi-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
