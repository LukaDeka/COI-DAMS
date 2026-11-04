# COI-DAMS: A Scalable Conflict of Interest Management and Analysis System

COI-DAMS is an open-source system for conferences for finding Conflicts of Interest (COIs) between the publishers/reviewers listed in a CSV file exported by conference management systems. It was developed as part of a bachelor thesis at TU Berlin (DAMS), which formalizes the algorithm and describes how it runs in a distributed manner. A link to the thesis will be added here after it's published.

Peer reviewers are often affiliated with the authors of the papers they review, which can cause a bias. The system builds a co-authorship graph from DBLP and, optionally, Semantic Scholar, and runs a distributed breadth-first search on Apache Spark to find COIs.


There are two types of COIs,
direct COIs, so pairs of people who have co-authored a piece of work,
and submarine COIs, so pairs who haven't necessarily worked together, but are connected through a path of co-authors.

A submarine COI doesn't necessarily indicate a real conflict, but it works as a heuristic. Each pair is scored with weights that depend on the path length.

How the COIs are scored, and how much submarine COIs influence the metric can be adjusted in the `.env` file. See `.env.example`.

## Installation
Install `uv`, then run `uv sync`.
That installs the `coi` and `qualify` commands into the project environment, which `uv run` puts on PATH.

## Getting data sources

`misc/fetch_data_sources.py` downloads and unpacks data sources the tool needs for the computation.

```bash
# Download/extract DBLP only
uv run python misc/fetch_data_sources.py

# Download/extract DBLP and Semantic Scholar (API-key needed)
# Apply for an API-key here: https://www.semanticscholar.org/product/api
export S2_API_KEY=...
uv run python misc/fetch_data_sources.py
```

DBLP always gets fetched. Semantic Scholar only gets downloaded when a key is provided via the `S2_API_KEY` environment variable or via the `--api-key` flag.

See `--help` for all flags.


## Usage

### Conflict of Interest detection (main feature)
Builds a graph of co-authors from DBLP, runs distributed BFS for the users from the `--users` CSV to find (submarine) COIs, and outputs each pair in a CSV file.

#### Running it locally
Omit the `--master` flag to run it local mode.
The amount of CPU cores, as well as memory can be set using the `--local-cores` and `--driver-memory` flags.

In this example the `--analysis` and `--unresolved-output` flags are used to output extra CSV files that contain information about the COIs, and which users weren't able to be disambiguated:
```bash
uv run coi \
  --dblp-xml data/dblp.xml --users input/Users.csv \
  --output output/coi.csv \
  --analysis --analysis-output output/coi-analysis.csv \
  --unresolved-output output/unresolved-users.csv \
  --max-depth 2 --local-cores 4 --driver-memory 8g
```

#### Running it on a cluster (`spark-submit`)

> [!NOTE]
> The dependencies (`pyspark`, `lxml`, `pyarrow`) need to be present in the workers' environment since `--py-files` only delivers the source files, not the dependencies

On a cluster, submit the job with `spark-submit` instead so the driver launches under Spark's own bootstrap, and
ship the package to every executor with `--py-files` (executors don't have the
project installed, only what's shipped alongside the job):

Example to submit the job on a cluster:
```bash
# Zip the source files (from inside src/, so coi_dams/ is at the zip root) to push them to the driver
(cd src && zip -r /tmp/coi_dams.zip coi_dams)

spark-submit \
  --master spark://10.0.0.0:7077 \
  --executor-memory 10g \
  --py-files /tmp/coi_dams.zip \
  src/coi_dams/main.py \
  --hdfs hdfs://10.0.0.0:9000 --hdfs-parquet hdfs://10.0.0.0:9000/coi/dblp.parquet \
  --webhdfs http://10.0.0.0:9870 \
  --shuffle-partitions 32 \
  --source-buckets 16 \
  --dblp-xml data/dblp.xml \
  --users input/Users.csv \
  --output output/coi.csv \
  --analysis --analysis-output output/coi-analysis.csv \
  --unresolved-output output/unresolved-users.csv \
  --max-depth 2
```
> [!NOTE]
>  Don't forget to re-zip `coi_dams.zip` whenever source code `src/coi_dams/*.py` is updated

For descriptions of the flags, see:
```bash
uv run coi --help
```

#### Adding Semantic Scholar as a second data source

Fetch the dumps as described in [Getting the data](#getting-the-data), then point both flags at them.
Both `coi` and `qualify` can optionally use Semantic Scholar alongside DBLP:

```bash
uv run coi \
  --dblp-xml data/dblp.xml --users input/Users.csv \
  --s2-authors-dir data/semantic_scholar/authors \
  --s2-papers-dir data/semantic_scholar/papers \
  --output output/coi.csv
```


> [!NOTE]
> The full S2 "papers" dataset is tens of GB (60 part files, ~1.5GB each at the time of writing).


### Qualify

The "qualify" feature, given an input list of users, outputs a list of qualified reviewers that have a certain amount of combined papers in the specified venues.
The required number of papers is `COI_REVIEWER_MIN_PAPERS`, and the venue list is `COI_QUALIFYING_VENUES`, both in the `.env` file (see `.env.example`).

Run it using e.g.:
```bash
uv run qualify --dblp-xml data/dblp.xml --output output/qualified-reviewers.csv --users input/Users.csv
```

Optionally use the `--reverse` flag to output only the unqualified reviewers.

For descriptions of the flags, see:
```bash
uv run qualify -h
```

The output includes these additional fields:

`total_qualifying_papers`: Sum of the papers published by the author in e.g. SIGMOD, VLDB, ICDE, EDBT, CIDR

`total_publication_count`: Number of distinct papers in DBLP by the author, counted across every alias they publish under

`disambiguation_level`: With values: 
* `certain`: The matching from the input user to the DBLP entity was done by a URL/ID
* `doubt`: The matching was done using heuristics (organization/country/author name)
* `failed`: All of the people with the same name were merged into one entity. These results can be wildly inaccurate

## Miscellaneous
`./misc/input_statistics.py` calculates how many rows (users) have the data about the URLs which is used for disambiguation.

