"""Explicit bounded Phase 4 model hyperparameters."""

from pathlib import Path

from pydantic import Field, model_validator

from michelin.config import StrictModel, load_yaml_mapping


class ModelingSettings(StrictModel):
    logistic_c: float = Field(gt=0)
    logistic_max_iter: int = Field(gt=0)
    logistic_tolerance: float = Field(gt=0)
    boosting_learning_rate: float = Field(gt=0, le=1)
    boosting_base_rounds: int = Field(gt=0)
    boosting_capacity_rounds: int = Field(gt=0)
    boosting_depth: int = Field(gt=0)
    prediction_rounds: int = Field(gt=0)
    calibration_c: float = Field(gt=0)
    calibration_bins: int = Field(ge=2)
    overfit_rows: int = Field(ge=2)
    watchlist_rows: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_capacity(self) -> "ModelingSettings":
        """Require the capacity experiment to add actual training rounds."""
        if self.boosting_capacity_rounds <= self.boosting_base_rounds:
            raise ValueError("boosting_capacity_rounds must exceed boosting_base_rounds")
        return self


def load_modeling_settings(path: Path) -> ModelingSettings:
    """Validate hyperparameters at the YAML input boundary."""
    return ModelingSettings.model_validate(load_yaml_mapping(path))
