#!/usr/bin/env python3

import csv

CSV_FILE = "input/Users.csv"

fields = {
    "Email": "Email",
    "Google Scholar URL": "Google Scholar URL",
    "Semantic Scholar URL": "Semantic Scholar URL",
    "DBLP URL": "DBLP URL",
    "OpenReview URL": "OpenReview Id",
}

counts = dict.fromkeys(fields, 0)
total = 0

# Counts for the three scholar-profile URLs
profile_counts = {
    3: 0,  # All 3
    2: 0,  # Any 2
    1: 0,  # Only 1
    0: 0,  # None
}

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        total += 1

        for label, column in fields.items():
            value = row.get(column, "")
            if value and value.strip():
                counts[label] += 1

        n_profiles = sum(
            bool((row.get(col, "") or "").strip())
            for col in (
                "Google Scholar URL",
                "Semantic Scholar URL",
                "DBLP URL",
            )
        )

        profile_counts[n_profiles] += 1

print(f"Total entries: {total:,}\n")

print("Coverage")
print("-" * 60)
for label, count in counts.items():
    pct = 100.0 * count / total if total else 0.0
    print(f"{label:22s} {count:8d} / {total:<8d} ({pct:6.2f}%)")

print("\nScholar profile coverage (Google Scholar + Semantic Scholar + DBLP)")
print("-" * 60)

labels = {
    3: "All 3",
    2: "Exactly 2",
    1: "Exactly 1",
    0: "None (0)",
}

for n in (3, 2, 1, 0):
    count = profile_counts[n]
    pct = 100.0 * count / total if total else 0.0
    print(f"{labels[n]:12s} {count:8d} / {total:<8d} ({pct:6.2f}%)")
