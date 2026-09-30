"""Google Places response models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class LocalizedText(StrictModel):
    text: str
    languageCode: str | None = None


class LatLng(StrictModel):
    latitude: float
    longitude: float


class Money(StrictModel):
    units: str | None = None


class PriceRange(StrictModel):
    startPrice: Money | None = None
    endPrice: Money | None = None


class OpeningPoint(StrictModel):
    day: int
    hour: int
    minute: int


class RegularOpeningPeriod(StrictModel):
    open: OpeningPoint
    close: OpeningPoint | None = None


class RegularOpeningHours(StrictModel):
    periods: tuple[RegularOpeningPeriod, ...] = ()
    weekdayDescriptions: tuple[str, ...] = ()


class CandidatePlace(StrictModel):
    id: str
    name: str | None = None
    formattedAddress: str | None = None
    location: LatLng | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("place id must not be blank")
        return value


class PlaceSearchResponse(StrictModel):
    places: tuple[CandidatePlace, ...] = ()
    nextPageToken: str | None = None


class PlaceDetails(StrictModel):
    id: str
    displayName: LocalizedText
    formattedAddress: str | None = None
    location: LatLng
    rating: float | None = None
    userRatingCount: int | None = None
    priceLevel: str | None = None
    priceRange: PriceRange | None = None
    businessStatus: str | None = None
    primaryType: str | None = None
    types: tuple[str, ...] = ()
    googleMapsUri: HttpUrl | None = None
    regularOpeningHours: RegularOpeningHours | None = None

    @field_validator("types", mode="before")
    @classmethod
    def validate_types(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError("types must be a list")
        return tuple(str(item) for item in value)
