import pandas as pd

from michelin.data.sources import normalize_guide_rows


def test_guide_normalization_retains_both_cities_without_collapsing_new_taipei() -> None:
    rows = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Location": ["Taipei, Taiwan", "New Taipei, Taiwan", "Taoyuan, Taiwan"],
        "Award": ["1 Star", "Bib Gourmand", "Selected Restaurants"],
        "Address": ["1 Road", "2 Road", "Taipei Restaurant, Taoyuan"],
    })
    result = normalize_guide_rows(rows, "2026-09-11", "michelin-my-maps")
    assert result[["name", "city"]].to_dict("records") == [
        {"name": "A", "city": "Taipei"}, {"name": "B", "city": "New Taipei"},
    ]
