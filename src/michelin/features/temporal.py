"""Label-agnostic spherical neighborhood counts over bounded reference batches."""

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EARTH_RADIUS_M = 6371008.8


def spherical_coordinates(rows: pd.DataFrame) -> np.ndarray:
    """Embed validated latitude/longitude on the unit sphere for exact chord queries."""
    coordinates = rows.loc[:, ["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any():
        raise ValueError("Density coordinates must be finite and inside latitude/longitude ranges")
    radians = np.deg2rad(coordinates)
    lat, lon = radians[:, 0], radians[:, 1]
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def neighborhood_density(query: pd.DataFrame, reference_batches: Iterable[pd.DataFrame], radius_m: float) -> pd.Series:
    """Index each streamed batch, count spherical neighbors, and exclude own identity."""
    if not math.isfinite(radius_m) or not 0 < radius_m <= math.pi * EARTH_RADIUS_M:
        raise ValueError("Density radius must be positive and no larger than half the Earth's circumference")
    if query["source_id"].isna().any() or query["source_id"].duplicated().any():
        raise ValueError("Density query requires unique non-null source identities")
    points = spherical_coordinates(query)
    chord = 2 * math.sin(radius_m / (2 * EARTH_RADIUS_M))
    counts = np.zeros(len(query), dtype=np.int64)
    query_ids = query["source_id"].to_numpy()
    found_self = np.zeros(len(query), dtype=np.int64)
    for batch in reference_batches:
        if batch.empty:
            continue
        if batch["source_id"].isna().any() or batch["source_id"].duplicated().any():
            raise ValueError("Density reference batch requires unique non-null source identities")
        reference_points = spherical_coordinates(batch)
        tree = cKDTree(reference_points)
        counts = counts + np.asarray(tree.query_ball_point(points, chord, return_length=True), dtype=np.int64)
        positions = {source_id: position for position, source_id in enumerate(batch["source_id"])}
        own = np.array([positions.get(source_id, -1) for source_id in query_ids])
        present = own >= 0
        if present.any():
            if not np.allclose(points[present], reference_points[own[present]], rtol=0, atol=1e-12):
                raise ValueError("Panel coordinates differ from frozen reference for the same identity")
            found_self = found_self + present.astype(np.int64)
            counts = counts - present.astype(np.int64)
    if not (found_self == 1).all():
        raise ValueError("Every panel identity must occur exactly once in the eligible density reference")
    return pd.Series(counts, index=pd.Index(query["restaurant_id"], name="restaurant_id"), name="neighborhood_density")
