"""Deterministic plain logistic skeleton with measured Phase 4 diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.tree import DecisionTreeClassifier

from michelin.config import AppConfig
from michelin.features.transforms import FittedTransforms, fit_transforms, transform_features
from michelin.model.dataset import ClassificationDataset, DatasetSplit, split_labels, validate_dataset_features, validate_split_class_counts, stratified_folds
from michelin.model.evaluate import CalibrationReport, EvaluationReport, evaluate_calibration, evaluate_probabilities, validate_identities
from michelin.model.thresholds import similar_to_starred
from michelin.model.settings import ModelingSettings, load_modeling_settings


@dataclass(frozen=True)
class FittedClassifier:
    transforms: FittedTransforms
    estimator: LogisticRegression


@dataclass(frozen=True)
class SkeletonDiagnostics:
    initial_probability: float
    initial_loss: float
    constant_probabilities: pd.Series
    overfit_loss: float
    overfit_labels_match: bool
    base_train_loss: float
    increased_capacity_train_loss: float
    watchlist_raw: pd.DataFrame
    watchlist_transformed: pd.DataFrame
    watchlist_probabilities: pd.DataFrame


@dataclass(frozen=True)
class SkeletonResult:
    model: FittedClassifier
    train_probabilities: pd.Series
    test_probabilities: pd.Series
    diagnostics: SkeletonDiagnostics


def initialized_logistic(
    matrix: pd.DataFrame, prior: float, seed: int, settings: ModelingSettings,
) -> LogisticRegression:
    """Create zero coefficients and a logit-prior intercept for warm-start training."""
    estimator = LogisticRegression(
        C=settings.logistic_c, max_iter=settings.logistic_max_iter,
        tol=settings.logistic_tolerance, solver="lbfgs", random_state=seed,
        class_weight=None, warm_start=True,
    )
    estimator.classes_ = np.array([0, 1])
    estimator.coef_ = np.zeros((1, matrix.shape[1]))
    estimator.intercept_ = np.array([np.log(prior / (1 - prior))])
    estimator.n_features_in_ = matrix.shape[1]
    estimator.feature_names_in_ = np.asarray(matrix.columns, dtype=object)
    return estimator


def predict_probabilities(model: FittedClassifier, raw_rows: pd.DataFrame) -> pd.Series:
    """Score each validated shared-feature row using frozen train-only transforms."""
    validate_identities(raw_rows, "Prediction rows")
    expected = {state.name for state in (*model.transforms.numeric, *model.transforms.categorical)}
    if raw_rows.columns.duplicated().any() or set(raw_rows.columns) != expected:
        raise ValueError("Prediction rows must contain exactly the shared model feature schema")
    matrix = transform_features(raw_rows, model.transforms)
    return pd.Series(model.estimator.predict_proba(matrix)[:, 1], index=raw_rows.index.copy(), name="probability")


def diagnostic_rows(labels: pd.Series, hard_negative: pd.Series, count: int) -> tuple[str, ...]:
    """Choose a deterministic class/slice watchlist without consulting holdout data."""
    candidates = (
        *labels.index[labels.eq(1)][:1],
        *labels.index[hard_negative][:1],
        *labels.index[labels.eq(0) & ~hard_negative][:1],
        *labels.index,
    )
    return tuple(dict.fromkeys(candidates))[:count]


def training_diagnostics(
    raw_train: pd.DataFrame, matrix: pd.DataFrame, labels: pd.Series,
    hard_negative: pd.Series, test_matrix: pd.DataFrame, initial_scores: np.ndarray,
    settings: ModelingSettings, seed: int,
) -> SkeletonDiagnostics:
    """Measure prior, fitted constant floor, tiny ceiling and staged capacity evidence."""
    prior = float(labels.mean())
    floor_train = pd.DataFrame(np.zeros(matrix.shape), index=matrix.index, columns=matrix.columns)
    floor_test = pd.DataFrame(np.zeros(test_matrix.shape), index=test_matrix.index, columns=matrix.columns)
    floor = initialized_logistic(floor_train, prior, seed, settings).fit(floor_train, labels)
    tiny_ids = diagnostic_rows(labels, hard_negative, settings.overfit_rows)
    tiny = matrix.loc[list(tiny_ids)]
    tiny_labels = labels.loc[tiny.index]
    ceiling = DecisionTreeClassifier(max_depth=None, random_state=seed).fit(tiny, tiny_labels)
    ceiling_scores = ceiling.predict_proba(tiny)[:, 1]
    booster = GradientBoostingClassifier(
        n_estimators=settings.boosting_capacity_rounds,
        learning_rate=settings.boosting_learning_rate,
        max_depth=settings.boosting_depth, random_state=seed,
        subsample=1.0,
    ).fit(matrix, labels)
    training_losses = tuple(
        float(log_loss(labels, scores[:, 1], labels=[0, 1]))
        for scores in booster.staged_predict_proba(matrix)
    )
    watch_ids = diagnostic_rows(labels, hard_negative, settings.watchlist_rows)
    watch_raw = raw_train.loc[list(watch_ids)].copy(deep=True)
    watch_matrix = matrix.loc[list(watch_ids)].copy(deep=True)
    watch_scores = pd.DataFrame({
        "prior": pd.Series(initial_scores, index=matrix.index).loc[list(watch_ids)].to_numpy(),
        **{f"boosting_round_{step}": scores[:, 1] for step, scores in enumerate(booster.staged_predict_proba(watch_matrix), start=1)},
    }, index=watch_matrix.index.copy())
    return SkeletonDiagnostics(
        initial_probability=float(initial_scores[0]),
        initial_loss=float(log_loss(labels, initial_scores, labels=[0, 1])),
        constant_probabilities=pd.Series(floor.predict_proba(floor_test)[:, 1], index=test_matrix.index.copy(), name="probability"),
        overfit_loss=float(log_loss(tiny_labels, ceiling_scores, labels=[0, 1])),
        overfit_labels_match=bool(np.array_equal(ceiling.predict(tiny), tiny_labels.to_numpy())),
        base_train_loss=training_losses[settings.boosting_base_rounds - 1],
        increased_capacity_train_loss=training_losses[-1],
        watchlist_raw=watch_raw, watchlist_transformed=watch_matrix,
        watchlist_probabilities=watch_scores,
    )


def train_skeleton(dataset: ClassificationDataset, split: DatasetSplit, config: AppConfig) -> SkeletonResult:
    """Fit only explicit training identities; leave holdout and display rows untouched."""
    validate_dataset_features(dataset.features, config)
    train_labels = split_labels(dataset, split.train_ids).loc[list(split.train_ids)]
    split_labels(dataset, split.test_ids)
    if set(split.train_ids).intersection(split.test_ids):
        raise ValueError("Training and holdout identities must be disjoint")
    if set((*split.train_ids, *split.test_ids)) != set(dataset.features.index):
        raise ValueError("Training and holdout identities must partition the complete modeling population")
    validate_split_class_counts(train_labels, 1, "Skeleton training partition")
    validate_identities(dataset.hard_negative, "Hard negatives")
    if set(dataset.hard_negative.index) != set(dataset.features.index):
        raise ValueError("Hard-negative identities must match modeling identities")
    if any(not isinstance(value, (bool, np.bool_)) for value in dataset.hard_negative):
        raise ValueError("Hard-negative flags must be non-null booleans")
    hard = dataset.hard_negative.loc[train_labels.index]
    if train_labels.loc[hard].ne(0).any():
        raise ValueError("Hard-negative training rows must have class 0 labels")
    if config.pipeline.models.explanation != "logistic_regression":
        raise ValueError("Phase 4 explanation skeleton requires logistic_regression")
    settings = load_modeling_settings(Path(__file__).resolve().parents[3] / "config/modeling.yaml")
    raw_train = dataset.features.loc[list(split.train_ids)]
    transforms = fit_transforms(raw_train, config.features)
    matrix = transform_features(raw_train, transforms)
    test_matrix = transform_features(dataset.features.loc[list(split.test_ids)], transforms)
    estimator = initialized_logistic(matrix, float(train_labels.mean()), config.pipeline.seed, settings)
    initial_scores = estimator.predict_proba(matrix)[:, 1]
    estimator.fit(matrix, train_labels)
    model = FittedClassifier(transforms, estimator)
    diagnostics = training_diagnostics(raw_train, matrix, train_labels, hard, test_matrix, initial_scores, settings, config.pipeline.seed)
    return SkeletonResult(
        model=model,
        train_probabilities=predict_probabilities(model, raw_train),
        test_probabilities=predict_probabilities(model, dataset.features.loc[list(split.test_ids)]),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class CalibratedClassifier:
    transforms: FittedTransforms
    estimator: GradientBoostingClassifier
    calibrator: LogisticRegression


@dataclass(frozen=True)
class FoldProvenance:
    validation_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationReport:
    global_importance: pd.Series
    contributions: pd.DataFrame
    top_five: dict[str, tuple[str, ...]]
    caveat: str


@dataclass(frozen=True)
class FeatureStage:
    features: tuple[str, ...]
    probabilities: pd.Series
    average_precision: float
    fold_average_precision: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationDiagnostics:
    stages: tuple[FeatureStage, ...]
    rationale: str
    schedule: str


@dataclass(frozen=True)
class CalibratedResult:
    explanation: FittedClassifier
    prediction: CalibratedClassifier
    holdout_probabilities: pd.Series
    holdout_evaluation: EvaluationReport
    oof_probabilities: pd.Series
    oof_evaluation: EvaluationReport
    oof_folds: tuple[FoldProvenance, ...]
    holdout_calibration: CalibrationReport
    oof_calibration: CalibrationReport
    cutoff: float
    final_explanation: FittedClassifier
    final_prediction: CalibratedClassifier
    diagnostics: CalibrationDiagnostics


def prediction_estimator(settings: ModelingSettings, seed: int) -> GradientBoostingClassifier:
    """Use fixed bounded capacity; never use holdout labels for early stopping."""
    return GradientBoostingClassifier(
        n_estimators=settings.prediction_rounds,
        learning_rate=settings.boosting_learning_rate,
        max_depth=settings.boosting_depth, subsample=1.0,
        random_state=seed, n_iter_no_change=None,
    )


def fit_explanation(dataset: ClassificationDataset, train_ids: tuple[str, ...], config: AppConfig) -> FittedClassifier:
    """Fit the plain LR explanation independently from the prediction model."""
    if config.pipeline.models.explanation != "logistic_regression":
        raise ValueError("Explanation model requires logistic_regression")
    labels = split_labels(dataset, train_ids).loc[list(train_ids)]
    validate_split_class_counts(labels, 1, "Explanation training partition")
    raw = dataset.features.loc[list(train_ids)]
    validate_dataset_features(raw, config)
    transforms = fit_transforms(raw, config.features)
    matrix = transform_features(raw, transforms)
    settings = load_modeling_settings(Path(__file__).resolve().parents[3] / "config/modeling.yaml")
    estimator = initialized_logistic(matrix, float(labels.mean()), config.pipeline.seed, settings).fit(matrix, labels)
    return FittedClassifier(transforms, estimator)


def fit_calibrated(dataset: ClassificationDataset, train_ids: tuple[str, ...], config: AppConfig) -> CalibratedClassifier:
    """Platt-fit inner OOF GBM decision logits, then refit the base GBM on train_ids."""
    if config.pipeline.models.prediction != "lightweight_gbm":
        raise ValueError("Prediction model requires lightweight_gbm")
    if config.pipeline.threshold.calibration != "platt_or_isotonic":
        raise ValueError("Calibration requires platt_or_isotonic resolved to Platt for this small-positive v1")
    labels = split_labels(dataset, train_ids).loc[list(train_ids)]
    raw = dataset.features.loc[list(train_ids)]
    validate_dataset_features(raw, config)
    settings = load_modeling_settings(Path(__file__).resolve().parents[3] / "config/modeling.yaml")
    folds = stratified_folds(dataset, train_ids, config.pipeline)
    inner_scores: list[pd.Series] = []
    for fold in folds:
        inner_raw = raw.loc[list(fold.train_ids)]
        transforms = fit_transforms(inner_raw, config.features)
        estimator = prediction_estimator(settings, config.pipeline.seed).fit(
            transform_features(inner_raw, transforms), labels.loc[list(fold.train_ids)],
        )
        logits = estimator.decision_function(transform_features(raw.loc[list(fold.test_ids)], transforms))
        inner_scores.append(pd.Series(logits, index=list(fold.test_ids)))
    calibration_input = pd.concat(inner_scores).loc[labels.index].to_numpy().reshape(-1, 1)
    calibrator = LogisticRegression(
        C=settings.calibration_c, max_iter=settings.logistic_max_iter,
        tol=settings.logistic_tolerance, solver="lbfgs", class_weight=None,
        random_state=config.pipeline.seed,
    ).fit(calibration_input, labels)
    transforms = fit_transforms(raw, config.features)
    estimator = prediction_estimator(settings, config.pipeline.seed).fit(transform_features(raw, transforms), labels)
    return CalibratedClassifier(transforms, estimator, calibrator)


def predict_calibrated(model: CalibratedClassifier, raw_rows: pd.DataFrame) -> pd.Series:
    """Apply sigmoid(a * GBM decision_function + b) with frozen train-only state."""
    validate_identities(raw_rows, "Calibrated prediction rows")
    expected = {state.name for state in (*model.transforms.numeric, *model.transforms.categorical)}
    if raw_rows.columns.duplicated().any() or set(raw_rows.columns) != expected:
        raise ValueError("Calibrated prediction rows must contain exactly the shared model feature schema")
    logits = model.estimator.decision_function(transform_features(raw_rows, model.transforms)).reshape(-1, 1)
    return pd.Series(model.calibrator.predict_proba(logits)[:, 1], index=raw_rows.index.copy(), name="probability")


def explain_rows(explanation: FittedClassifier, raw_rows: pd.DataFrame) -> ExplanationReport:
    """Report signed LR log-odds contributions; these are not GBM attributions."""
    predict_probabilities(explanation, raw_rows)
    matrix = transform_features(raw_rows, explanation.transforms)
    coefficients = pd.Series(explanation.estimator.coef_[0], index=matrix.columns, name="coefficient")
    contributions = matrix.mul(coefficients, axis="columns")
    return ExplanationReport(
        global_importance=coefficients, contributions=contributions,
        top_five={identity: tuple(row.abs().sort_values(ascending=False, kind="stable").index[:5]) for identity, row in contributions.iterrows()},
        caveat="Logistic explanation model contributions describe over-representation / correlation, not causation. They explain logistic log-odds, not the distinct prediction GBM's probabilities.",
    )


def training_feature_stages(dataset: ClassificationDataset, train_ids: tuple[str, ...], config: AppConfig, settings: ModelingSettings) -> tuple[FeatureStage, ...]:
    """Compare fixed GBM groups with raw fold-local transforms on training rows only."""
    groups = (("city",), ("average_rating", "review_count"), ("neighborhood_density",), ("price_range",), ("cuisine",))
    folds = stratified_folds(dataset, train_ids, config.pipeline)
    stages: list[FeatureStage] = []
    for step in range(1, len(groups) + 1):
        features = tuple(name for group in groups[:step] for name in group)
        predictions: list[pd.Series] = []
        fold_metrics: list[float] = []
        for fold in folds:
            raw = dataset.features.loc[list(fold.train_ids)]
            transforms = fit_transforms(raw, config.features)
            columns = [name for name in transforms.feature_names if name in features or any(name.startswith(feature + "__") for feature in features)]
            estimator = prediction_estimator(settings, config.pipeline.seed).fit(
                transform_features(raw, transforms).loc[:, columns], dataset.labels.loc[list(fold.train_ids)],
            )
            scores = pd.Series(estimator.predict_proba(transform_features(dataset.features.loc[list(fold.test_ids)], transforms).loc[:, columns])[:, 1], index=list(fold.test_ids), name="probability")
            predictions.append(scores)
            fold_metrics.append(float(average_precision_score(dataset.labels.loc[scores.index], scores)))
        probabilities = pd.concat(predictions).reindex(list(train_ids))
        stages.append(FeatureStage(features, probabilities, float(average_precision_score(dataset.labels.loc[probabilities.index], probabilities)), tuple(fold_metrics)))
    return tuple(stages)


def evaluate_dataset_rows(dataset: ClassificationDataset, probabilities: pd.Series, cutoff: float) -> EvaluationReport:
    """Evaluate identities in the supplied honest scoring population only."""
    identities = probabilities.index
    labels = dataset.labels.loc[identities]
    return evaluate_probabilities(labels, probabilities, dataset.hard_negative.loc[identities], dataset.features.loc[identities, "city"], cutoff, max(1, int(labels.sum())))


def train_calibrated(dataset: ClassificationDataset, split: DatasetSplit, config: AppConfig) -> CalibratedResult:
    """Evaluate train-only models first, then honest nested OOF and distinct full refits."""
    if set(split.train_ids).intersection(split.test_ids) or set((*split.train_ids, *split.test_ids)) != set(dataset.features.index):
        raise ValueError("Training and holdout identities must disjointly partition the modeling population")
    split_labels(dataset, split.test_ids)
    cutoff = config.pipeline.threshold.similar_to_starred_cutoff
    if cutoff != 0.2:  #experimentation: fixed cutoff 20% (previously 50%).
        raise ValueError("Phase 5 requires the approved fixed Similar-to-Starred cutoff 0.2")  #experimentation: fixed cutoff 20%.
    settings = load_modeling_settings(Path(__file__).resolve().parents[3] / "config/modeling.yaml")
    explanation = fit_explanation(dataset, split.train_ids, config)
    prediction = fit_calibrated(dataset, split.train_ids, config)
    holdout = predict_calibrated(prediction, dataset.features.loc[list(split.test_ids)])
    similar_to_starred(holdout, cutoff)
    holdout_evaluation = evaluate_dataset_rows(dataset, holdout, cutoff)
    holdout_calibration = evaluate_calibration(dataset.labels.loc[holdout.index], holdout, settings.calibration_bins)
    diagnostics = CalibrationDiagnostics(
        stages=training_feature_stages(dataset, split.train_ids, config, settings),
        rationale="Platt uses one slope and intercept fitted on inner OOF decision logits; few positives do not support flexible isotonic calibration.",
        schedule=f"Fixed learning_rate={settings.boosting_learning_rate}, n_estimators={settings.prediction_rounds}, product={settings.boosting_learning_rate * settings.prediction_rounds}; early stopping disabled (n_iter_no_change=None). No holdout-based tuning.",
    )
    all_ids = tuple(dataset.features.index)
    folds = stratified_folds(dataset, all_ids, config.pipeline)
    predictions = tuple(predict_calibrated(fit_calibrated(dataset, fold.train_ids, config), dataset.features.loc[list(fold.test_ids)]) for fold in folds)
    oof = pd.concat(predictions).reindex(dataset.features.index)
    oof_evaluation = evaluate_dataset_rows(dataset, oof, cutoff)
    return CalibratedResult(
        explanation=explanation, prediction=prediction,
        holdout_probabilities=holdout, holdout_evaluation=holdout_evaluation,
        oof_probabilities=oof, oof_evaluation=oof_evaluation,
        oof_folds=tuple(FoldProvenance(fold.test_ids, fold.train_ids, fold.train_ids) for fold in folds),
        holdout_calibration=holdout_calibration,
        oof_calibration=evaluate_calibration(dataset.labels.loc[oof.index], oof, settings.calibration_bins),
        cutoff=cutoff, final_explanation=fit_explanation(dataset, all_ids, config),
        final_prediction=fit_calibrated(dataset, all_ids, config), diagnostics=diagnostics,
    )
