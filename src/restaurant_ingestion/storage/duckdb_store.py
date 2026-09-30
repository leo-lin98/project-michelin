"""DuckDB storage layer for restaurant ingestion."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from restaurant_ingestion.models import PlaceDetails, PriceRange, RegularOpeningPeriod


class DuckDbStore:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(database_path))

    def close(self) -> None:
        self._connection.close()

    def create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurants (
                place_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                address VARCHAR,
                latitude DOUBLE NOT NULL,
                longitude DOUBLE NOT NULL,
                rating DOUBLE,
                review_count INTEGER,
                price_level VARCHAR,
                price_range VARCHAR,
                website VARCHAR,
                phone VARCHAR,
                google_maps_url VARCHAR,
                business_status VARCHAR,
                primary_category VARCHAR,
                is_michelin BOOLEAN NOT NULL,
                last_updated TIMESTAMP NOT NULL
            )
            """
        )
        # CREATE TABLE IF NOT EXISTS is a no-op on databases created before a column
        # was added, so columns added later must be applied explicitly for an existing
        # database to converge on the same schema as a fresh one.
        self._connection.execute("ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS price_range VARCHAR")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurant_categories (
                place_id VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                PRIMARY KEY (place_id, category)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurant_hours (
                place_id VARCHAR NOT NULL,
                day_of_week INTEGER NOT NULL,
                open_time VARCHAR NOT NULL,
                close_time VARCHAR NOT NULL,
                PRIMARY KEY (place_id, day_of_week, open_time, close_time)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS restaurant_michelin (
                place_id VARCHAR PRIMARY KEY,
                michelin_name VARCHAR NOT NULL,
                michelin_category VARCHAR NOT NULL,
                michelin_stars INTEGER NOT NULL,
                michelin_bib_gourmand BOOLEAN NOT NULL,
                guide_year INTEGER,
                last_updated TIMESTAMP NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS place_raw_responses (
                place_id VARCHAR PRIMARY KEY,
                response_json JSON NOT NULL,
                response_hash VARCHAR NOT NULL,
                fetched_at TIMESTAMP NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS place_raw_response_history (
                place_id VARCHAR NOT NULL,
                response_json JSON NOT NULL,
                response_hash VARCHAR NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                PRIMARY KEY (place_id, response_hash, fetched_at)
            )
            """
        )
        self._connection.execute(
            """
            INSERT INTO place_raw_response_history (place_id, response_json, response_hash, fetched_at)
            SELECT place_id, response_json, response_hash, fetched_at FROM place_raw_responses
            ON CONFLICT DO NOTHING
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
                checkpoint_id VARCHAR PRIMARY KEY,
                checkpoint_type VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                payload_json JSON NOT NULL,
                page_token VARCHAR,
                updated_at TIMESTAMP NOT NULL,
                error_message VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_places (
                place_id VARCHAR PRIMARY KEY,
                discovered_at TIMESTAMP NOT NULL,
                source_checkpoint_id VARCHAR NOT NULL,
                enriched_at TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS api_call_log (
                call_id VARCHAR PRIMARY KEY,
                endpoint VARCHAR NOT NULL,
                sku_family VARCHAR NOT NULL,
                status_code INTEGER NOT NULL,
                retry_count INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                place_id VARCHAR,
                checkpoint_id VARCHAR
            )
            """
        )

    def upsert_candidate(self, place_id: str, checkpoint_id: str) -> None:
        self._connection.execute(
            """
            INSERT INTO candidate_places (place_id, discovered_at, source_checkpoint_id, enriched_at)
            VALUES (?, ?, ?, NULL)
            ON CONFLICT (place_id) DO NOTHING
            """,
            [place_id, now_utc(), checkpoint_id],
        )

    def unenriched_candidate_ids(self, limit: int, stale_after_days: int) -> list[str]:
        stale_cutoff = now_utc() - timedelta(days=stale_after_days)
        rows = self._connection.execute(
            """
            SELECT candidate_places.place_id
            FROM candidate_places
            LEFT JOIN restaurants USING (place_id)
            WHERE candidate_places.enriched_at IS NULL
               OR restaurants.last_updated IS NULL
               OR restaurants.last_updated < ?
            ORDER BY candidate_places.discovered_at, candidate_places.place_id
            LIMIT ?
            """,
            [stale_cutoff, limit],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def stale_or_missing_place_ids(self, place_ids: tuple[str, ...], stale_after_days: int) -> list[str]:
        unique_place_ids = tuple(dict.fromkeys(place_ids))
        if not unique_place_ids:
            return []

        stale_cutoff = now_utc() - timedelta(days=stale_after_days)
        values_clause = ",".join(["(?)"] * len(unique_place_ids))
        rows = self._connection.execute(
            f"""
            WITH requested(place_id) AS (VALUES {values_clause})
            SELECT requested.place_id
            FROM requested
            LEFT JOIN restaurants USING (place_id)
            WHERE restaurants.last_updated IS NULL
               OR restaurants.last_updated < ?
            ORDER BY requested.place_id
            """,
            [*unique_place_ids, stale_cutoff],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def sample_restaurant_price_state(self, sample_size: int, seed: int) -> list[tuple[str, str | None]]:
        """Deterministically sample (place_id, current price_level) rows for a backfill run.

        The sample size and seed are interpolated because DuckDB does not accept bind
        parameters inside a SAMPLE clause; both are coerced to int to keep the clause
        free of caller-supplied text.
        """
        rows = self._connection.execute(
            f"""
            SELECT place_id, price_level
            FROM (
                SELECT place_id, price_level
                FROM restaurants
                USING SAMPLE reservoir({int(sample_size)} ROWS) REPEATABLE ({int(seed)})
            )
            ORDER BY place_id
            """
        ).fetchall()
        return [(str(row[0]), None if row[1] is None else str(row[1])) for row in rows]

    def missing_price_state(self, limit: int, retry_after_days: int) -> list[tuple[str, str | None]]:
        """Select restaurants still missing a price field, least recently refreshed first.

        Rows refreshed inside `retry_after_days` are excluded so repeated runs walk
        forward through untried rows instead of re-paying for rows a previous run
        already found empty. Ordering by `last_updated` means the stalest data, which
        is the likeliest to have gained a price since it was fetched, is tried first.
        """
        retry_cutoff = now_utc() - timedelta(days=retry_after_days)
        rows = self._connection.execute(
            """
            SELECT place_id, price_level
            FROM restaurants
            WHERE (price_level IS NULL OR price_range IS NULL)
              AND last_updated < ?
            ORDER BY last_updated, place_id
            LIMIT ?
            """,
            [retry_cutoff, limit],
        ).fetchall()
        return [(str(row[0]), None if row[1] is None else str(row[1])) for row in rows]

    def enriched_restaurant_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM restaurants WHERE is_michelin = false").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def upsert_restaurant(self, place: PlaceDetails, raw_response_json: str, response_hash: str, is_michelin: bool) -> None:
        updated_at = now_utc()
        self._connection.execute(
            """
            INSERT INTO restaurants (
                place_id, name, address, latitude, longitude, rating, review_count, price_level, price_range,
                website, phone, google_maps_url, business_status, primary_category, is_michelin, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)
            ON CONFLICT (place_id) DO UPDATE SET
                name = excluded.name,
                address = excluded.address,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                rating = excluded.rating,
                review_count = excluded.review_count,
                price_level = excluded.price_level,
                price_range = excluded.price_range,
                google_maps_url = excluded.google_maps_url,
                business_status = excluded.business_status,
                primary_category = excluded.primary_category,
                is_michelin = restaurants.is_michelin OR excluded.is_michelin,
                last_updated = excluded.last_updated
            """,
            [
                place.id,
                place.displayName.text,
                place.formattedAddress,
                place.location.latitude,
                place.location.longitude,
                place.rating,
                place.userRatingCount,
                place.priceLevel,
                format_price_range(place.priceRange),
                str(place.googleMapsUri) if place.googleMapsUri is not None else None,
                place.businessStatus,
                place.primaryType,
                is_michelin,
                updated_at,
            ],
        )
        self.replace_categories(place.id, place.types)
        self.replace_hours(place.id, place.regularOpeningHours.periods if place.regularOpeningHours is not None else ())
        self.upsert_raw_response(place.id, raw_response_json, response_hash, updated_at)
        self.mark_candidate_enriched(place.id, updated_at)

    def replace_categories(self, place_id: str, categories: tuple[str, ...]) -> None:
        self._connection.execute("DELETE FROM restaurant_categories WHERE place_id = ?", [place_id])
        for category in categories:
            self._connection.execute(
                "INSERT INTO restaurant_categories (place_id, category) VALUES (?, ?) ON CONFLICT DO NOTHING",
                [place_id, category],
            )

    def replace_hours(self, place_id: str, periods: tuple[RegularOpeningPeriod, ...]) -> None:
        self._connection.execute("DELETE FROM restaurant_hours WHERE place_id = ?", [place_id])
        for period in periods:
            close_time = point_time(period.close) if period.close is not None else "unknown"
            self._connection.execute(
                """
                INSERT INTO restaurant_hours (place_id, day_of_week, open_time, close_time)
                VALUES (?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [place_id, period.open.day, point_time(period.open), close_time],
            )

    def upsert_raw_response(self, place_id: str, response_json: str, response_hash: str, fetched_at: datetime) -> None:
        """Append the collected response and update its latest-value projection atomically."""
        parameters = [place_id, response_json, response_hash, fetched_at]
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.execute(
                """
                INSERT INTO place_raw_response_history (place_id, response_json, response_hash, fetched_at)
                VALUES (?, CAST(? AS JSON), ?, ?)
                ON CONFLICT DO NOTHING
                """, parameters,
            )
            self._connection.execute(
                """
                INSERT INTO place_raw_responses (place_id, response_json, response_hash, fetched_at)
                VALUES (?, CAST(? AS JSON), ?, ?)
                ON CONFLICT (place_id) DO UPDATE SET
                    response_json = excluded.response_json,
                    response_hash = excluded.response_hash,
                    fetched_at = excluded.fetched_at
                """, parameters,
            )
            self._connection.execute("COMMIT")
        except duckdb.Error:
            self._connection.execute("ROLLBACK")
            raise

    def mark_candidate_enriched(self, place_id: str, enriched_at: datetime) -> None:
        self._connection.execute("UPDATE candidate_places SET enriched_at = ? WHERE place_id = ?", [enriched_at, place_id])

    def upsert_michelin_metadata(
        self,
        place_id: str,
        michelin_name: str,
        michelin_category: str,
        michelin_stars: int,
        michelin_bib_gourmand: bool,
        guide_year: int | None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO restaurant_michelin (
                place_id, michelin_name, michelin_category, michelin_stars,
                michelin_bib_gourmand, guide_year, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (place_id) DO UPDATE SET
                michelin_name = excluded.michelin_name,
                michelin_category = excluded.michelin_category,
                michelin_stars = excluded.michelin_stars,
                michelin_bib_gourmand = excluded.michelin_bib_gourmand,
                guide_year = excluded.guide_year,
                last_updated = excluded.last_updated
            """,
            [place_id, michelin_name, michelin_category, michelin_stars, michelin_bib_gourmand, guide_year, now_utc()],
        )
        self._connection.execute("UPDATE restaurants SET is_michelin = true WHERE place_id = ?", [place_id])

    def sync_existing_michelin_metadata(
        self, rows: tuple[tuple[str, str, str, int, bool, int | None], ...],
    ) -> int:
        """Atomically synchronize labels and flags; leave Google fields and timestamps untouched."""
        if not rows:
            return 0
        place_ids = [row[0] for row in rows]
        existing = {row[0] for row in self._connection.execute(
            "SELECT place_id FROM restaurants WHERE place_id IN (SELECT unnest(?))", [place_ids]
        ).fetchall()}
        matched = tuple(row for row in rows if row[0] in existing)
        if not matched:
            return 0
        values_clause = ",".join("(?, ?, ?, ?, ?, ?, ?)" for _ in matched)
        updated_at = now_utc()
        parameters = [value for row in matched for value in (*row, updated_at)]
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.execute(
                f"""
                INSERT INTO restaurant_michelin (
                    place_id, michelin_name, michelin_category, michelin_stars,
                    michelin_bib_gourmand, guide_year, last_updated
                ) VALUES {values_clause}
                ON CONFLICT (place_id) DO UPDATE SET
                    michelin_name = excluded.michelin_name,
                    michelin_category = excluded.michelin_category,
                    michelin_stars = excluded.michelin_stars,
                    michelin_bib_gourmand = excluded.michelin_bib_gourmand,
                    guide_year = excluded.guide_year,
                    last_updated = excluded.last_updated
                WHERE (restaurant_michelin.michelin_name, restaurant_michelin.michelin_category,
                       restaurant_michelin.michelin_stars, restaurant_michelin.michelin_bib_gourmand,
                       restaurant_michelin.guide_year)
                    IS DISTINCT FROM (excluded.michelin_name, excluded.michelin_category,
                                      excluded.michelin_stars, excluded.michelin_bib_gourmand,
                                      excluded.guide_year)
                """, parameters,
            )
            self._connection.execute(
                "UPDATE restaurants SET is_michelin = true WHERE place_id IN (SELECT unnest(?)) AND NOT is_michelin",
                [[row[0] for row in matched]],
            )
            self._connection.execute("COMMIT")
        except duckdb.Error:
            self._connection.execute("ROLLBACK")
            raise
        return len(matched)

    def upsert_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_type: str,
        status: str,
        payload_json: str,
        page_token: str | None,
        error_message: str | None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO ingestion_checkpoints (
                checkpoint_id, checkpoint_type, status, payload_json, page_token, updated_at, error_message
            )
            VALUES (?, ?, ?, CAST(? AS JSON), ?, ?, ?)
            ON CONFLICT (checkpoint_id) DO UPDATE SET
                status = excluded.status,
                payload_json = excluded.payload_json,
                page_token = excluded.page_token,
                updated_at = excluded.updated_at,
                error_message = excluded.error_message
            """,
            [checkpoint_id, checkpoint_type, status, payload_json, page_token, now_utc(), error_message],
        )

    def checkpoint_status(self, checkpoint_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT status FROM ingestion_checkpoints WHERE checkpoint_id = ?",
            [checkpoint_id],
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def log_api_call(
        self,
        call_id: str,
        endpoint: str,
        sku_family: str,
        status_code: int,
        retry_count: int,
        place_id: str | None,
        checkpoint_id: str | None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO api_call_log (
                call_id, endpoint, sku_family, status_code, retry_count, created_at, place_id, checkpoint_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (call_id) DO NOTHING
            """,
            [call_id, endpoint, sku_family, status_code, retry_count, now_utc(), place_id, checkpoint_id],
        )

    def api_call_count(self, sku_family: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM api_call_log WHERE sku_family = ?",
            [sku_family],
        ).fetchone()
        if row is None:
            return 0
        return int(row[0])


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def point_time(point: Any) -> str:
    return f"{int(point.hour):02d}:{int(point.minute):02d}"


def format_price_range(price_range: PriceRange | None) -> str | None:
    """Render a Places priceRange as "<start>-<end>" or "<start>+" in bare currency units.

    An absent upper bound is open-ended, so a start price of 2000 becomes "2000+".
    Return None for absent ranges, missing lower bounds, or bounds without units.
    """
    if price_range is None:
        return None
    start_price = price_range.startPrice
    end_price = price_range.endPrice
    if start_price is None or start_price.units is None:
        return None
    if end_price is None:
        return f"{start_price.units}+"
    if end_price.units is None:
        return None
    return f"{start_price.units}-{end_price.units}"
