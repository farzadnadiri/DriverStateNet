"""Nested participant-grouped classification and mixed-effects behavioural models."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
# Small, declared grids per model family; the inner loop picks one value per outer fold.
GRIDS = {
    "logreg": C_GRID,
    "linear_svm": C_GRID,
    "gbm": [(2, 0.05), (2, 0.1), (3, 0.05), (3, 0.1)],  # (max_depth, learning_rate)
}


def _pipeline(param, model: str = "logreg"):
    if model == "logreg":
        estimator = LogisticRegression(C=param, class_weight="balanced", max_iter=5000)
    elif model == "linear_svm":
        estimator = LinearSVC(C=param, class_weight="balanced", max_iter=20000)
    elif model == "gbm":
        depth, rate = param
        estimator = HistGradientBoostingClassifier(
            max_depth=depth, learning_rate=rate, max_iter=200, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        )
    else:
        raise ValueError(model)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _uar(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def _select_c(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, model: str = "logreg"
) -> tuple[object, dict[object, float]]:
    """Leave-one-fold-out over the training folds; returns the grid value with best mean inner UAR."""
    scores = {}
    for param in GRIDS[model]:
        fold_scores = []
        for inner in np.unique(folds):
            train, val = folds != inner, folds == inner
            if len(np.unique(y[train])) < len(np.unique(y)) or len(np.unique(y[val])) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = _pipeline(param, model).fit(x[train], y[train])
            fold_scores.append(_uar(y[val], fitted.predict(x[val])))
        scores[param] = float(np.mean(fold_scores)) if fold_scores else np.nan
    best = max(scores, key=lambda p: -np.inf if np.isnan(scores[p]) else scores[p])
    return best, scores


def nested_cv(
    data: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    fold_column: str = "fixed_fold",
    model: str = "logreg",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer = fixed participant folds; inner = leave-one-fold-out on the training folds.

    Returns (out-of-fold predictions, per-outer-fold selection log). No preprocessing or
    hyperparameter choice ever sees the outer test fold.
    """
    x = data[feature_columns].to_numpy(float)
    y = data[target].to_numpy()
    folds = data[fold_column].to_numpy()
    predictions = []
    log = []
    for outer in np.unique(folds):
        train, test = folds != outer, folds == outer
        best_c, inner_scores = _select_c(x[train], y[train], folds[train], model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = _pipeline(best_c, model).fit(x[train], y[train])
        frame = data.loc[test, ["participant_name", "scenario_path", fold_column]].copy()
        frame["true_label"] = y[test]
        frame["pred_label"] = fitted.predict(x[test])
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(x[test])
            for i, cls in enumerate(fitted.classes_):
                frame[f"proba__{cls}"] = proba[:, i]
        predictions.append(frame)
        log.append(
            {
                "outer_fold": outer,
                "best_C": str(best_c),
                "inner_uar": inner_scores[best_c],
                "outer_uar": _uar(y[test], frame["pred_label"]),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
            }
        )
    out = pd.concat(predictions).reset_index(drop=True)
    out["true_label"] = out["true_label"].astype(str)
    out["pred_label"] = out["pred_label"].astype(str)
    return out, pd.DataFrame(log)


def fit_and_apply(
    train: pd.DataFrame, apply: pd.DataFrame, feature_columns: list[str], target: str, fold_column: str = "fixed_fold"
) -> pd.DataFrame:
    """Tune C by leave-one-fold-out on `train`, refit on all of it, predict `apply`."""
    x, y = train[feature_columns].to_numpy(float), train[target].to_numpy()
    best_c, _ = _select_c(x, y, train[fold_column].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _pipeline(best_c).fit(x, y)
    out = apply[["participant_name", "scenario_path"]].copy()
    out["true_label"] = apply[target].astype(str).to_numpy()
    out["pred_label"] = model.predict(apply[feature_columns].to_numpy(float)).astype(str)
    return out


def holm(p_values: pd.Series) -> pd.Series:
    order = p_values.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adjusted, 1.0), index=order.index).reindex(p_values.index)


def mixed_effects(
    data: pd.DataFrame,
    measures: list[str],
    formula_rhs: str,
    terms: list[str],
    standardise_on: pd.Series | None,
    re_formula: str | None = None,
) -> pd.DataFrame:
    """One random-intercept LMM per measure. With `standardise_on` (sober R1 trips), outcomes
    are z-scored on those rows so coefficients are in reference-SD units; with None they stay
    in raw units (used for reaction times in log seconds and crash probability)."""
    import statsmodels.formula.api as smf

    rows = []
    for measure in measures:
        frame = data.dropna(subset=[measure]).copy()
        if standardise_on is None:
            frame["z"] = frame[measure]
        else:
            ref = frame.loc[standardise_on.reindex(frame.index).fillna(False).astype(bool), measure]
            frame["z"] = (frame[measure] - ref.mean()) / ref.std()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # re_formula="~cd" adds a per-participant random slope for the CD effect
            fit = smf.mixedlm(f"z ~ {formula_rhs}", frame, groups=frame["participant_name"],
                              re_formula=re_formula).fit(reml=True)
            null = smf.mixedlm("z ~ 1", frame, groups=frame["participant_name"]).fit(reml=True)
        between = float(null.cov_re.iloc[0, 0])
        icc = between / (between + float(null.scale))
        ci = fit.conf_int()
        for term in terms:
            if term not in fit.params.index:
                continue
            rows.append(
                {
                    "measure": measure,
                    "term": term,
                    "units": "raw" if standardise_on is None else "reference_sd",
                    "estimate_sd": float(fit.params[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                    "p_value": float(fit.pvalues[term]),
                    "se": float(fit.bse[term]),
                    # sensitivity: smallest true effect detectable with 80% power at two-sided
                    # alpha = .05 given this design's standard error (1.96 + 0.84 = 2.80 SE)
                    "mde_80": float(2.80 * fit.bse[term]),
                    "n_trips": int(len(frame)),
                    "n_participants": int(frame["participant_name"].nunique()),
                    "icc_null_model": icc,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm_within_term"] = result.groupby("term")["p_value"].transform(holm)
    return result
