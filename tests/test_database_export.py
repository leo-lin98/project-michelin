import csv
from hashlib import sha256
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from michelin.config import load_config
from michelin.data import database
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def export_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "restaurants.duckdb"
    store = DuckDbStore(db_path)
    store.create_schema()
    store.close()
    with duckdb.connect(str(db_path)) as connection:
        for place_id, city, category in [
            ("star", "Taipei City", "japanese_restaurant"),
            ("bib", "New Taipei City", "chinese_restaurant"),
            ("ordinary-1", "Taipei City", "noodle_shop"),
            ("ordinary-2", "New Taipei City", "restaurant"),
            ("ordinary-3", "New Taipei City", "restaurant"),
            ("outside", "Taoyuan City", "restaurant"),
        ]:
            connection.execute(
                """INSERT INTO restaurants
                (place_id,name,address,latitude,longitude,rating,review_count,price_level,primary_category,is_michelin,last_updated)
                VALUES (?, ?, ?, 25.0, 121.5, 4.5, 100, ?, ?, ?, ?)""",
                [place_id, f"Google {place_id}", f"1 Road, {city}, Taiwan",
                 "PRICE_LEVEL_MODERATE" if place_id == "star" else None,
                 category, place_id == "star", datetime(2026, 9, 11)],
            )
            raw = json.dumps({
                "id": place_id, "displayName": {"text": f"Google {place_id}"},
                "formattedAddress": f"1 Road, {city}, Taiwan", "location": {"latitude": 25.0, "longitude": 121.5},
                "rating": 4.5, "userRatingCount": 100, "primaryType": category,
                "priceLevel": "PRICE_LEVEL_MODERATE" if place_id == "star" else None,
                "businessStatus": "OPERATIONAL",
            }, sort_keys=True, separators=(",", ":"))
            connection.execute(
                "INSERT INTO place_raw_responses VALUES (?, ?, ?, ?)",
                [place_id, raw, sha256(raw.encode()).hexdigest(), datetime(2026, 8, 6)],
            )
        connection.execute("UPDATE restaurants SET business_status='OPERATIONAL'")
        connection.execute("INSERT INTO restaurant_michelin VALUES ('star','Guide star','1 Star',1,false,NULL,?)", [datetime(2026, 9, 11)])
    store = DuckDbStore(db_path)
    store.create_schema()
    store.close()
    guide_path = tmp_path / "guide.csv"
    with guide_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Location", "Award", "Url"])
        writer.writerow(["Guide star", "Taipei, Taiwan", "1 Star", "https://guide.example/star"])
        writer.writerow(["Guide bib", "New Taipei, Taiwan", "Bib Gourmand", "https://guide.example/bib"])
    mapping_path = tmp_path / "resolved.csv"
    with mapping_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["place_id", "michelin_url"])
        writer.writerow(["star", "https://guide.example/star"])
        writer.writerow(["bib", "https://guide.example/bib"])
    return db_path, guide_path, mapping_path


def test_export_uses_verified_id_join_google_features_and_real_collection_dates(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={"pools": config.pools.model_copy(update={"ordinary_in_sample_limit": 2, "ordinary_out_of_sample_limit": 1})})
    before = db_path.read_bytes()
    result = database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    panel = pd.read_csv(tmp_path / "output" / "processed" / "labeled_restaurants.csv").set_index("source_id")

    assert result["rows"] == 5
    assert result["starred_rows"] == 1
    assert result["hard_negative_rows"] == 1
    assert panel.loc["bib", "class"] == 0
    assert panel.loc["bib", "is_hard_negative"]
    assert panel.loc["bib", "group"] == "in_sample"
    assert panel.loc["bib", "city"] == "New Taipei"
    assert panel.loc["star", "price_level"] == 2
    assert panel.loc["star", "cuisine"] == "japanese_restaurant"
    assert panel.loc["star", "feature_snapshot_date"].startswith("2026-08-06")
    assert panel.loc["star", "name"] == "Google star"
    assert pd.isna(panel.loc["bib", "price_level"])
    assert "outside" not in panel.index
    assert db_path.read_bytes() == before
    path = tmp_path / "output" / "processed" / "labeled_restaurants.csv"
    before_panel = path.read_bytes()
    database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    assert path.read_bytes() == before_panel


def test_export_blocks_missing_new_taipei_guide_mapping_and_writes_actionable_report(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    mapping_path.write_text("place_id,michelin_url\nstar,https://guide.example/star\n")
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    with pytest.raises(ValueError, match="Guide coverage incomplete"):
        database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    report = pd.read_csv(tmp_path / "output" / "interim" / "guide_coverage.csv")
    assert report.loc[report["name"] == "Guide bib", "status"].item() == "missing_place_id"
    assert not (tmp_path / "output" / "processed" / "labeled_restaurants.csv").exists()


def test_explicit_unresolved_exclusion_is_audited_and_allows_build(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    mapping_path.write_text("place_id,michelin_url\nstar,https://guide.example/star\n")
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("DELETE FROM restaurants WHERE place_id='bib'")
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={
        "eligibility": config.eligibility.model_copy(update={"unresolved_guide_exclusions": ("https://guide.example/bib",)}),
        "pools": config.pools.model_copy(update={"ordinary_in_sample_limit": 2, "ordinary_out_of_sample_limit": 1}),
    })
    result = database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    coverage = pd.read_csv(tmp_path / "output" / "interim" / "guide_coverage.csv")
    assert coverage.loc[coverage["name"].eq("Guide bib"), "status"].item() == "excluded_by_user"
    assert result["rows"] == 4
    assert result["hard_negative_rows"] == 0
    mapping_path.write_text("place_id,michelin_url\nstar,https://guide.example/star\nbib,https://guide.example/bib\n")
    with pytest.raises(ValueError, match="now has a mapping"):
        database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "recheck", config)


@pytest.mark.parametrize("address,expected", [
    ("1 Road, Taipei City, Taiwan", "Taipei"),
    ("1 Road, New Taipei City, Taiwan", "New Taipei"),
    ("220台灣新北市板橋區", "New Taipei"),
    ("106台灣臺北市大安區", "Taipei"),
    ("1 Road, Taoyuan City, Taiwan", None),
    ("Taipei restaurant, Taoyuan City, Taiwan", None),
    ("1 Road, Wenshan District, 台灣 Taiwan 116", None),
])
def test_city_classification_never_guesses_from_restaurant_names(address: str, expected: str | None) -> None:
    assert database.city_from_address(address) == expected


def test_export_routes_bad_ordinary_features_to_dlq_and_still_fills_pools(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            """INSERT INTO restaurants
            (place_id,name,address,latitude,longitude,rating,review_count,price_level,primary_category,is_michelin,last_updated)
            VALUES ('invalid','Invalid','1 Road, New Taipei City, Taiwan',25.0,121.5,7.0,100,NULL,'restaurant',false,?)""",
            [datetime(2026, 9, 11)],
        )
        connection.execute("INSERT INTO place_raw_responses VALUES ('invalid','{}','hash',?)", [datetime(2026, 9, 11)])
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={"pools": config.pools.model_copy(update={"ordinary_in_sample_limit": 2, "ordinary_out_of_sample_limit": 1})})
    result = database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    assert result["rows"] == 5
    dlq = pd.read_csv(tmp_path / "output" / "interim" / "database_dlq.csv").set_index("source_id")
    assert "average_rating" in dlq.loc["invalid", "reason"]


def test_export_refuses_guide_identity_without_database_record(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("DELETE FROM restaurants WHERE place_id = 'bib'")
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    with pytest.raises(ValueError, match="Guide coverage incomplete"):
        database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    report = pd.read_csv(tmp_path / "output" / "interim" / "guide_coverage.csv")
    assert report.loc[report["name"] == "Guide bib", "status"].item() == "missing_database_features"


def test_eligibility_excludes_closed_guide_and_ordinary_rows_with_audit(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("UPDATE restaurants SET business_status='OPERATIONAL'")
        connection.execute("UPDATE restaurants SET business_status='CLOSED_TEMPORARILY' WHERE place_id IN ('bib','ordinary-3')")
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={"eligibility": config.eligibility.model_copy(update={"business_statuses": ("OPERATIONAL",)})})
    config = config.model_copy(update={"pools": config.pools.model_copy(update={"ordinary_in_sample_limit": 1, "ordinary_out_of_sample_limit": 1})})
    result = database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    assert result["rows"] == 3
    assert result["hard_negative_rows"] == 0
    dlq = pd.read_csv(tmp_path / "output" / "interim" / "database_dlq.csv").set_index("source_id")
    assert dlq.loc["bib", "reason"] == "excluded_business_status:CLOSED_TEMPORARILY"
    assert dlq.loc["ordinary-3", "reason"] == "excluded_business_status:CLOSED_TEMPORARILY"
    coverage = pd.read_csv(tmp_path / "output" / "interim" / "guide_coverage.csv").set_index("place_id")
    assert coverage.loc["bib", "status"] == "excluded_business_status"
    audit = (tmp_path / "output" / "interim" / "missingness_audit.json").read_text()
    assert '"final_panel"' in audit
    assert '"published_reconciliation": "deferred"' in audit


@pytest.mark.parametrize("status", ["OPERATIONAL", "CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY", "FUTURE_OPENING", None])
def test_current_policy_keeps_every_status_in_guide_and_ordinary_pools(tmp_path: Path, status: str | None) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("UPDATE restaurants SET business_status=?", [status])
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={"pools": config.pools.model_copy(update={"ordinary_in_sample_limit": 2, "ordinary_out_of_sample_limit": 1})})
    result = database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    assert result["rows"] == 5
    assert result["hard_negative_rows"] == 1
    assert result["starred_rows"] == 1
    coverage = pd.read_csv(tmp_path / "output" / "interim" / "guide_coverage.csv")
    assert coverage["status"].eq("ready").all()


def test_blocked_export_still_audits_available_eligible_data(tmp_path: Path) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("UPDATE restaurants SET business_status='OPERATIONAL'")
    mapping_path.write_text("place_id,michelin_url\nstar,https://guide.example/star\n")
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    with pytest.raises(ValueError, match="Guide coverage incomplete"):
        database.export_database_panel(db_path, guide_path, mapping_path, tmp_path / "output", config)
    audit = (tmp_path / "output" / "interim" / "missingness_audit.json").read_text()
    assert '"final_panel": null' in audit
    assert '"ordinary_candidate"' in audit
