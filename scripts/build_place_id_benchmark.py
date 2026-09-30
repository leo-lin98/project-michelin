"""Build the frozen place_id benchmark manifest from the Michelin source guide.

This selects the benchmark rows and emits the manifest + empty (PENDING) label files.
It deliberately does NOT assign expected outcomes: labels must be set by hand against the
real, fetched Google payloads (see scripts/capture_benchmark_fixtures.py) so they are
justified by evidence rather than fitted to the matching implementation.
"""

from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "raw" / "guide" / "michelin_my_maps.csv"
BENCHMARK_DIR = ROOT / "tests" / "benchmark"
MANIFEST_CSV = BENCHMARK_DIR / "place_id_benchmark.csv"
TUNE_LABELS_CSV = BENCHMARK_DIR / "labels.tune.csv"
HELDOUT_LABELS_CSV = BENCHMARK_DIR / "labels.heldout.csv"

MANIFEST_COLUMNS = (
    "row_id",
    "category",
    "slice",
    "name",
    "award",
    "address",
    "location",
    "latitude",
    "longitude",
    "phone_number",
    "website_url",
    "michelin_url",
)
LABEL_COLUMNS = ("row_id", "name", "expected_outcome", "expected_place_id", "evidence")
PENDING = "PENDING_FIXTURE"

# (name, award, category, slice). Selection is the benchmark definition, not matching logic.
SELECTION: tuple[tuple[str, str, str, str], ...] = (
    ("JUNTO", "Selected Restaurants", "base_unresolved", "tune"),
    ("fumée", "Selected Restaurants", "base_unresolved", "tune"),
    ("A Cut", "1 Star", "base_unresolved", "tune"),
    ("Unnamed Clay Oven Roll", "Bib Gourmand", "base_unresolved", "tune"),
    ("Lin Ju", "Selected Restaurants", "base_unresolved", "tune"),
    ("Page", "Selected Restaurants", "base_unresolved", "tune"),
    ("Yuu", "Selected Restaurants", "base_unresolved", "heldout"),
    ("Rong Ju", "Selected Restaurants", "base_unresolved", "heldout"),
    ("Circum-", "1 Star", "base_unresolved", "heldout"),
    ("3927", "Selected Restaurants", "base_unresolved", "heldout"),
    ("Yu Yu 1969", "Selected Restaurants", "base_unresolved", "heldout"),
    ("KUR", "Selected Restaurants", "base_unresolved", "heldout"),
    ("Sushi Kajin", "1 Star", "resolved_clean", "tune"),
    ("Eika", "2 Stars", "resolved_clean", "tune"),
    ("Taïrroir", "3 Stars", "resolved_clean", "heldout"),
    ("NOBUO", "1 Star", "resolved_clean", "heldout"),
    ("Bencotto", "Selected Restaurants", "same_building", "tune"),
    ("Ya Ge", "1 Star", "same_building", "heldout"),
    ("A", "2 Stars", "short_generic_name", "tune"),
    ("Lin", "Selected Restaurants", "short_generic_name", "heldout"),
)


def slug(value: str) -> str:
    lowered = value.casefold()
    kept = "".join(character if character.isalnum() else "-" for character in lowered)
    return "-".join(part for part in kept.split("-") if part)


def load_source_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_source_row(rows: list[dict[str, str]], name: str, award: str) -> dict[str, str]:
    matches = [row for row in rows if row["Name"] == name and row["Award"] == award]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source row for name={name!r} award={award!r}, found {len(matches)}")
    return matches[0]


def build_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for index, (name, award, category, slice_name) in enumerate(SELECTION, start=1):
        source = find_source_row(rows, name, award)
        manifest_rows.append(
            {
                "row_id": f"{index:02d}-{slug(name)}",
                "category": category,
                "slice": slice_name,
                "name": name,
                "award": award,
                "address": source["Address"],
                "location": source["Location"],
                "latitude": source["Latitude"],
                "longitude": source["Longitude"],
                "phone_number": source.get("PhoneNumber", ""),
                "website_url": source.get("WebsiteUrl", ""),
                "michelin_url": source.get("Url", ""),
            }
        )
    return manifest_rows


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def main() -> None:
    rows = load_source_rows()
    manifest_rows = build_manifest_rows(rows)
    write_csv(MANIFEST_CSV, MANIFEST_COLUMNS, manifest_rows)

    label_rows = [
        {"row_id": row["row_id"], "name": row["name"], "expected_outcome": PENDING, "expected_place_id": "", "evidence": ""}
        for row in manifest_rows
    ]
    write_csv(TUNE_LABELS_CSV, LABEL_COLUMNS, [row for row, source in zip(label_rows, manifest_rows) if source["slice"] == "tune"])
    write_csv(HELDOUT_LABELS_CSV, LABEL_COLUMNS, [row for row, source in zip(label_rows, manifest_rows) if source["slice"] == "heldout"])

    tune = sum(1 for row in manifest_rows if row["slice"] == "tune")
    heldout = sum(1 for row in manifest_rows if row["slice"] == "heldout")
    print({"manifest_rows": len(manifest_rows), "tune": tune, "heldout": heldout, "manifest": str(MANIFEST_CSV)})


if __name__ == "__main__":
    sys.exit(main())
