"""Developer training checks derived from split and schema requirements."""

from pathlib import Path

import pandas as pd
import pytest

from michelin.config import load_config
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.train import train_skeleton


@pytest.mark.parametrize("split", [DatasetSplit(("a", "b"), ("b", "c")), DatasetSplit(("a",), ("b",))])
def test_invalid_split_never_trains(split: DatasetSplit) -> None:
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))
    ids = pd.Index(["a", "b", "c", "d"], name="restaurant_id")
    rows = pd.DataFrame({"average_rating": [3.0] * 4, "review_count": [5] * 4, "neighborhood_density": [1] * 4, "price_range": ["1-200"] * 4, "cuisine": ["restaurant"] * 4, "city": ["Taipei"] * 4}, index=ids)
    labels = pd.Series([0, 1, 0, 1], index=ids)
    dataset = ClassificationDataset(rows, labels, labels.eq(0), rows.iloc[:0])
    with pytest.raises(ValueError):
        train_skeleton(dataset, split, config)
