import pandas as pd

from michelin.data.identity import IdentityMatchConfig, resolve_guide_ordinary_duplicates


def test_resolve_guide_ordinary_duplicates_removes_matching_ordinary_row() -> None:
    guide_rows = pd.DataFrame(
        [
            {
                "source_id": "guide-1",
                "name": "Logy",
                "address": "1 Taipei Road",
                "latitude": 25.0330,
                "longitude": 121.5654,
            }
        ]
    )
    ordinary_rows = pd.DataFrame(
        [
            {
                "source_id": "osm-1",
                "name": "Logy",
                "address": "1 Taipei Road",
                "latitude": 25.0331,
                "longitude": 121.5655,
            },
            {
                "source_id": "osm-2",
                "name": "Neighborhood Noodles",
                "address": "2 Taipei Road",
                "latitude": 25.02,
                "longitude": 121.52,
            },
        ]
    )
    config = IdentityMatchConfig(name_similarity_threshold=0.92, distance_threshold_meters=50.0)

    result = resolve_guide_ordinary_duplicates(guide_rows, ordinary_rows, config)

    assert result.ordinary_without_guide_duplicates["source_id"].tolist() == ["osm-2"]
    assert result.alias_table.loc[0, "guide_source_id"] == "guide-1"
