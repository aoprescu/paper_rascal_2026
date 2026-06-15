"""
=============================================================================
PIPELINE v4 -- Repeated Stratified 5-Fold CV by hospital volume group
=============================================================================
Design change:
  - Hospital grouping is based on real care activity instead of the previous
    hospital category.
  - For each hospital, the mean "Altas CMBD validas" over 2020-2022 is computed.
  - Hospitals are split into LowVolume / HighVolume using the median mean volume.

Strategies:
  - Local: local train fold only.
  - Hybrid: same volume group hospitals, excluding X, plus local train fold.
  - External: same volume group hospitals, excluding X.
  - BigData: all hospitals, excluding X.
  - BigDataHybrid: all hospitals, excluding X, plus local train fold.

Inside each CV fold:
  1. Split the strategy-specific train set into model_train (85%) + cal_set (15%).
  2. Train XGBoost on model_train.
  3. Fit Platt scaling on cal_set.
  4. Estimate thresholds (Youden, F1, F2) on cal_set.
  5. Evaluate on test_fold with pre-determined thresholds.

Requires:
  pip install pandas numpy scikit-learn xgboost scipy openpyxl
=============================================================================
"""

import os
import re
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "INSERT_PATH_TO_PICKLE_HERE"
OUTPUT_DIR = "INSERT_PATH_TO_OUTPUT_DIR_HERE"
SEED = 42

VOLUME_GROUPS_CSV = "hospital_volume_groups_2020_2022.csv"

# CV configuration
N_FOLDS = 5
N_REPEATS = 5
CAL_SIZE = 0.15
MIN_POSITIVE_FOLD = 3

# Holdout sensitivity analysis
HOLDOUT_TEST_SIZE = 0.25

TARGET = "hospital_outcome"
HOSPITAL_COL = "centro"

STRATEGIES = ["Local", "External", "Hybrid", "BigData", "BigDataHybrid"]
VOLUME_GROUP_COL = "volume_group"

INCLUDED_HOSPITALS = [
    "H.U. Virgen del Rocio",
    "H.U. Virgen de las Nieves",
    "H.U. Virgen de la Victoria",
    "H.U. Reina Sofia",
    "H.U. San Cecilio",
    "H.U. Virgen Macarena",
    "H.U. Regional de Malaga",
    "H.U. Torrecardenas",
    "H.U. Virgen de Valme",
    "H.U. de Jaen",
    "H.U. Juan Ramon Jimenez",
    "H.U. de Jerez de la Frontera",
    "H. Infanta Margarita",
    "H.U. Puerta del Mar",
    "H. Infanta Elena",
    "H. San Juan de la Cruz",
    "H.U. de Puerto Real",
    "H. La Merced",
    "H. Punta de Europa",
    "H. San Agustin",
    "H. La Inmaculada",
    "H. Santa Ana",
    "H. de la Serrania",
    "H. de Antequera",
    "H. Valle de los Pedroches",
    "H. de La Linea de la Concepcion",
    "H. de La Axarquia",
    "H. de Baza",
    "H. de Riotinto",
]

DROP_FEATURES = [
    TARGET,
    HOSPITAL_COL,
    VOLUME_GROUP_COL,
    "es_estancia_unica",
    "num_shots", # aporta la misma información que la variable is_vaccinated
    "lab_lymphocyte_percentage_val",
    "lab_lymphocyte_percentage_below",
    "lab_lymphocyte_percentage_over",
]

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


# =============================================================================
# HOSPITAL VOLUME GROUPS
# =============================================================================

def _strip_accents(text):
    text = "" if pd.isna(text) else str(text)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def extract_hospital_code(value):
    """Extract a leading five-digit hospital code when present."""
    match = re.match(r"^\s*(\d{5})", "" if pd.isna(value) else str(value))
    return match.group(1) if match else None


def normalize_hospital_name(value):
    """Normalize hospital labels for fallback matching when no code is present."""
    text = _strip_accents(value).lower()
    text = re.sub(r"^\s*\d{5}\s*-\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_allowed_hospital_names():
    return {normalize_hospital_name(h) for h in INCLUDED_HOSPITALS}


def normalize_colname(value):
    text = _strip_accents(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_cmbd_volume_file(path, year):
    """Read one CMBD Excel and return hospital/year valid discharge counts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CMBD file not found for {year}: {path}")

    df = pd.read_excel(path, sheet_name="Altas Hospital", header=3)
    col_lookup = {normalize_colname(c): c for c in df.columns}

    hospital_col = None
    valid_col = None
    for norm, original in col_lookup.items():
        if norm == "hospital":
            hospital_col = original
        if "altas" in norm and "cmbd" in norm and "valid" in norm:
            valid_col = original

    if hospital_col is None or valid_col is None:
        raise ValueError(
            f"Could not find Hospital / Altas CMBD validas columns in {path}"
        )

    out = df[[hospital_col, valid_col]].copy()
    out.columns = ["hospital_label", "altas_cmbd_validas"]
    out["hospital_label"] = out["hospital_label"].astype(str).str.strip()
    out["hospital_code"] = out["hospital_label"].apply(extract_hospital_code)
    out["hospital_name_norm"] = out["hospital_label"].apply(normalize_hospital_name)
    out["altas_cmbd_validas"] = pd.to_numeric(
        out["altas_cmbd_validas"], errors="coerce"
    )
    out["year"] = year
    out = out.dropna(subset=["altas_cmbd_validas"])
    out = out[out["hospital_label"].ne("") & out["hospital_code"].notna()]
    return out[
        [
            "year",
            "hospital_code",
            "hospital_label",
            "hospital_name_norm",
            "altas_cmbd_validas",
        ]
    ]


def load_hospital_volume_groups(volume_files):
    """Compute LowVolume / HighVolume groups from 2020-2022 CMBD activity."""
    records = pd.concat(
        [read_cmbd_volume_file(path, year) for year, path in volume_files.items()],
        ignore_index=True,
    )

    # Aggregate by hospital code to handle denomination changes across years.
    grouped = []
    for code, g in records.groupby("hospital_code", sort=False):
        latest = g.sort_values("year").iloc[-1]
        row = {
            "hospital_code": code,
            "hospital_name": latest["hospital_label"],
            "hospital_name_norm": latest["hospital_name_norm"],
            "n_years": int(g["year"].nunique()),
            "mean_altas_cmbd_validas": float(g["altas_cmbd_validas"].mean()),
        }
        for year in sorted(volume_files):
            vals = g.loc[g["year"] == year, "altas_cmbd_validas"]
            row[f"altas_{year}"] = float(vals.iloc[0]) if len(vals) else np.nan
        grouped.append(row)

    volume_df = pd.DataFrame(grouped)
    allowed_names = make_allowed_hospital_names()
    volume_df = volume_df[volume_df["hospital_name_norm"].isin(allowed_names)].copy()
    missing = sorted(allowed_names - set(volume_df["hospital_name_norm"]))
    if missing:
        raise ValueError(
            "These INCLUDED_HOSPITALS were not found in the CMBD files: "
            + ", ".join(missing)
        )

    volume_df = volume_df.sort_values("mean_altas_cmbd_validas").reset_index(drop=True)
    median_cutoff = float(volume_df["mean_altas_cmbd_validas"].median())
    volume_df["median_cutoff"] = median_cutoff
    volume_df[VOLUME_GROUP_COL] = np.where(
        volume_df["mean_altas_cmbd_validas"] >= median_cutoff,
        "HighVolume",
        "LowVolume",
    )

    lookup = {}
    for _, row in volume_df.iterrows():
        lookup[f"code:{row['hospital_code']}"] = row.to_dict()
        lookup[f"name:{row['hospital_name_norm']}"] = row.to_dict()

    return volume_df, lookup, median_cutoff


def load_hospital_volume_groups_csv(path):
    """Load the curated 29-hospital volume classification CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volume groups CSV not found: {path}")

    volume_df = pd.read_csv(path)
    required = {
        "hospital_code",
        "hospital_name",
        "mean_altas_cmbd_validas",
        "median_cutoff",
        VOLUME_GROUP_COL,
    }
    missing_cols = sorted(required - set(volume_df.columns))
    if missing_cols:
        raise ValueError(
            f"Volume groups CSV is missing columns: {', '.join(missing_cols)}"
        )

    volume_df["hospital_code"] = (
        volume_df["hospital_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    volume_df["hospital_name_norm"] = volume_df["hospital_name"].apply(
        normalize_hospital_name
    )
    volume_df["mean_altas_cmbd_validas"] = pd.to_numeric(
        volume_df["mean_altas_cmbd_validas"], errors="raise"
    )
    volume_df["median_cutoff"] = pd.to_numeric(
        volume_df["median_cutoff"], errors="raise"
    )

    allowed_names = make_allowed_hospital_names()
    volume_df = volume_df[volume_df["hospital_name_norm"].isin(allowed_names)].copy()
    missing = sorted(allowed_names - set(volume_df["hospital_name_norm"]))
    if missing:
        raise ValueError(
            "These INCLUDED_HOSPITALS were not found in the curated volume CSV: "
            + ", ".join(missing)
        )
    if len(volume_df) != len(INCLUDED_HOSPITALS):
        raise ValueError(
            f"Expected {len(INCLUDED_HOSPITALS)} hospitals in volume CSV, "
            f"found {len(volume_df)} after filtering."
        )

    median_cutoff = float(volume_df["median_cutoff"].iloc[0])
    lookup = {}
    for _, row in volume_df.iterrows():
        row_dict = row.to_dict()
        lookup[f"code:{row['hospital_code']}"] = row_dict
        lookup[f"name:{row['hospital_name_norm']}"] = row_dict

    return volume_df, lookup, median_cutoff


def get_volume_record(hospital_value, volume_lookup):
    code = extract_hospital_code(hospital_value)
    if code and f"code:{code}" in volume_lookup:
        return volume_lookup[f"code:{code}"]

    name_key = f"name:{normalize_hospital_name(hospital_value)}"
    return volume_lookup.get(name_key)


def attach_volume_groups(df, volume_lookup):
    """Attach volume group to each patient row using the hospital column."""
    df = df.copy()
    allowed_names = make_allowed_hospital_names()
    df = df[
        df[HOSPITAL_COL].apply(lambda hosp: normalize_hospital_name(hosp) in allowed_names)
    ].copy()
    hospital_records = {}
    missing = []

    for hosp in sorted(df[HOSPITAL_COL].dropna().unique()):
        rec = get_volume_record(hosp, volume_lookup)
        if rec is None:
            missing.append(hosp)
        else:
            hospital_records[hosp] = rec

    if missing:
        print("\nWARNING: hospitals without CMBD volume match; they will be skipped:")
        for hosp in missing:
            print(f"  - {hosp}")

    df[VOLUME_GROUP_COL] = df[HOSPITAL_COL].map(
        lambda hosp: hospital_records.get(hosp, {}).get(VOLUME_GROUP_COL, np.nan)
    )

    matched = pd.DataFrame(
        [
            {
                "dataset_hospital": hosp,
                "hospital_code": rec["hospital_code"],
                "cmbd_hospital_name": rec["hospital_name"],
                "n_years_volume": rec["n_years"],
                "mean_altas_cmbd_validas": rec["mean_altas_cmbd_validas"],
                "median_cutoff": rec["median_cutoff"],
                VOLUME_GROUP_COL: rec[VOLUME_GROUP_COL],
            }
            for hosp, rec in hospital_records.items()
        ]
    ).sort_values("dataset_hospital")

    return df, matched


def validate_volume_assignments(df, matched):
    """Fail early if a known sentinel hospital has an unexpected volume group."""
    checks = {
        "H. La Inmaculada": "LowVolume",
        "H. San Agustin": "LowVolume",
    }
    for hospital_name, expected_group in checks.items():
        norm_name = normalize_hospital_name(hospital_name)
        rows = matched[
            matched["dataset_hospital"].apply(normalize_hospital_name) == norm_name
        ]
        if rows.empty:
            continue
        observed = sorted(rows[VOLUME_GROUP_COL].dropna().unique())
        if observed != [expected_group]:
            raise ValueError(
                f"{hospital_name} must be {expected_group}, but matched as {observed}. "
                "Check that you are running the updated volume-group pipeline and not "
                "an older output/script."
            )


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    df = df.copy()
    if "lab_neutrophil_val" in df.columns and "lab_lymphocyte_val" in df.columns:
        df["nlr"] = df["lab_neutrophil_val"] / df["lab_lymphocyte_val"].replace(0, np.nan)
        df.loc[df["nlr"] > 100, "nlr"] = np.nan
    if "lab_urea_val" in df.columns and "lab_creatinine_val" in df.columns:
        df["bun_cr_ratio"] = df["lab_urea_val"] / df["lab_creatinine_val"].replace(0, np.nan)
        df.loc[df["bun_cr_ratio"] > 200, "bun_cr_ratio"] = np.nan
    if "lab_ast_val" in df.columns and "lab_alt_val" in df.columns:
        df["deritis_ratio"] = df["lab_ast_val"] / df["lab_alt_val"].replace(0, np.nan)
        df.loc[df["deritis_ratio"] > 50, "deritis_ratio"] = np.nan
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_FEATURES]


# =============================================================================
# MODEL + CALIBRATION
# =============================================================================

def train_calibrate_within_fold(X_train_fold, y_train_fold):
    """
    Within one CV fold:
    1. Split train_fold into model_train (85%) + cal_set (15%).
    2. Train XGBoost on model_train.
    3. Fit Platt scaling on cal_set.
    4. Estimate thresholds on cal_set only.

    Returns: (model, platt, threshold_youden, threshold_f1, threshold_f2)
    """
    y_arr = y_train_fold.values if hasattr(y_train_fold, "values") else np.array(y_train_fold)

    # Very small hospitals/sets: skip the internal calibration split.
    if len(X_train_fold) < 60 or y_arr.sum() < 6:
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_fold, y_train_fold, verbose=False)
        yp = model.predict_proba(X_train_fold)[:, 1]
        return (
            model,
            None,
            _find_threshold(y_arr, yp, "youden"),
            _find_threshold(y_arr, yp, "f1"),
            _find_threshold(y_arr, yp, "f2"),
        )

    try:
        X_model, X_cal, y_model, y_cal = train_test_split(
            X_train_fold,
            y_train_fold,
            test_size=CAL_SIZE,
            random_state=SEED,
            stratify=y_train_fold,
        )
    except ValueError:
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_fold, y_train_fold, verbose=False)
        yp = model.predict_proba(X_train_fold)[:, 1]
        return (
            model,
            None,
            _find_threshold(y_arr, yp, "youden"),
            _find_threshold(y_arr, yp, "f1"),
            _find_threshold(y_arr, yp, "f2"),
        )

    model = XGBClassifier(**XGBOOST_PARAMS)
    y_model_arr = y_model.values if hasattr(y_model, "values") else np.array(y_model)

    if len(X_model) > 80 and y_model_arr.sum() >= 5:
        try:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_model,
                y_model,
                test_size=0.15,
                random_state=SEED,
                stratify=y_model,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        except ValueError:
            model.fit(X_model, y_model, verbose=False)
    else:
        model.fit(X_model, y_model, verbose=False)

    y_proba_cal = model.predict_proba(X_cal)[:, 1]
    y_cal_arr = y_cal.values if hasattr(y_cal, "values") else np.array(y_cal)

    platt = LogisticRegression(random_state=SEED)
    platt.fit(y_proba_cal.reshape(-1, 1), y_cal_arr)
    y_proba_cal_platt = platt.predict_proba(y_proba_cal.reshape(-1, 1))[:, 1]

    t_youden = _find_threshold(y_cal_arr, y_proba_cal_platt, "youden")
    t_f1 = _find_threshold(y_cal_arr, y_proba_cal_platt, "f1")
    t_f2 = _find_threshold(y_cal_arr, y_proba_cal_platt, "f2")

    return model, platt, t_youden, t_f1, t_f2


def _find_threshold(y_true, y_proba, method="youden"):
    """Optimal threshold. Call only with train/cal data, never test data."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    if method == "youden":
        fpr, tpr, th = roc_curve(y_true, y_proba)
        return th[np.argmax(tpr - fpr)]
    if method == "f1":
        ths = np.arange(0.05, 0.90, 0.01)
        f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in ths]
        return ths[np.argmax(f1s)]
    if method == "f2":
        ths = np.arange(0.05, 0.90, 0.01)
        f2s = [fbeta_score(y_true, (y_proba >= t).astype(int), beta=2) for t in ths]
        return ths[np.argmax(f2s)]
    return 0.5


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_fold(y_true, y_proba_raw, platt, t_youden, t_f1, t_f2):
    """Evaluate on a test fold with thresholds estimated before seeing the test."""
    if len(np.unique(y_true)) < 2:
        return None

    res = {}
    res["auroc"] = roc_auc_score(y_true, y_proba_raw)
    res["auprc"] = average_precision_score(y_true, y_proba_raw)
    res["brier_raw"] = brier_score_loss(y_true, y_proba_raw)

    if platt is not None:
        y_proba = platt.predict_proba(y_proba_raw.reshape(-1, 1))[:, 1]
    else:
        y_proba = y_proba_raw
    res["brier_cal"] = brier_score_loss(y_true, y_proba)

    res["f1_t05_raw"] = f1_score(y_true, (y_proba_raw >= 0.5).astype(int))
    res["f1_t05_cal"] = f1_score(y_true, (y_proba >= 0.5).astype(int))
    res["f2_t05_cal"] = fbeta_score(y_true, (y_proba >= 0.5).astype(int), beta=2)

    yp_y = (y_proba >= t_youden).astype(int)
    res["threshold_youden"] = t_youden
    res["f1_youden"] = f1_score(y_true, yp_y)
    res["f2_youden"] = fbeta_score(y_true, yp_y, beta=2)
    tn, fp, fn, tp = confusion_matrix(y_true, yp_y, labels=[0, 1]).ravel()
    res["recall_youden"] = recall_score(y_true, yp_y)
    res["precision_youden"] = precision_score(y_true, yp_y, zero_division=0)
    res["specificity_youden"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    yp_f1 = (y_proba >= t_f1).astype(int)
    res["threshold_f1"] = t_f1
    res["f1_optcal"] = f1_score(y_true, yp_f1)

    yp_f2 = (y_proba >= t_f2).astype(int)
    res["threshold_f2"] = t_f2
    res["f2_optcal"] = fbeta_score(y_true, yp_f2, beta=2)
    res["recall_f2"] = recall_score(y_true, yp_f2)
    res["precision_f2"] = precision_score(y_true, yp_f2, zero_division=0)

    res["n_test"] = len(y_true)
    res["n_pos_test"] = int(y_true.sum())
    return res


# =============================================================================
# TRAIN SET CONSTRUCTION
# =============================================================================

def build_train_cv(strategy, df, hosp, x_train_fold, y_train_fold, feat_cols):
    """
    Build the strategy-specific train set.
    x_train_fold / y_train_fold are the local data inside the outer CV fold.
    """
    if strategy == "Local":
        return x_train_fold.copy(), y_train_fold.copy()

    if strategy in ["External", "Hybrid"]:
        volume_group = df.loc[df[HOSPITAL_COL] == hosp, VOLUME_GROUP_COL].iloc[0]
        other = df[
            (df[VOLUME_GROUP_COL] == volume_group)
            & (df[HOSPITAL_COL] != hosp)
        ]
    elif strategy in ["BigData", "BigDataHybrid"]:
        other = df[df[HOSPITAL_COL] != hosp]
    else:
        return None, None

    other_X = other[[c for c in feat_cols if c in other.columns]]
    other_y = other[TARGET]

    if strategy in ["External", "BigData"]:
        return other_X, other_y

    return (
        pd.concat([x_train_fold, other_X], axis=0),
        pd.concat([y_train_fold, other_y], axis=0),
    )


# =============================================================================
# REPORTING HELPERS
# =============================================================================

METRICS_TO_AGG = [
    "auroc",
    "auprc",
    "brier_raw",
    "brier_cal",
    "f1_t05_raw",
    "f1_t05_cal",
    "f1_youden",
    "f1_optcal",
    "f2_t05_cal",
    "f2_youden",
    "f2_optcal",
    "recall_youden",
    "precision_youden",
    "specificity_youden",
    "recall_f2",
    "precision_f2",
]


def aggregate_cv_results(df_cv):
    return df_cv.groupby(["hospital", VOLUME_GROUP_COL, "strategy"]).agg(
        **{f"{m}_mean": (m, "mean") for m in METRICS_TO_AGG},
        **{f"{m}_sd": (m, "std") for m in METRICS_TO_AGG},
        n_folds=("auroc", "count"),
    ).reset_index()


def make_strategy_summary(df_agg):
    rows = []
    for strat, df_s in df_agg.groupby("strategy"):
        row = {"strategy": strat, "n_hospitals": len(df_s)}
        for m in METRICS_TO_AGG:
            row[f"{m}_mean_across_hospitals"] = df_s[f"{m}_mean"].mean()
            row[f"{m}_sd_across_hospitals"] = df_s[f"{m}_mean"].std()
        rows.append(row)
    return pd.DataFrame(rows)


def make_volume_strategy_summary(df_agg):
    rows = []
    for (vol, strat), df_s in df_agg.groupby([VOLUME_GROUP_COL, "strategy"]):
        row = {VOLUME_GROUP_COL: vol, "strategy": strat, "n_hospitals": len(df_s)}
        for m in METRICS_TO_AGG:
            row[f"{m}_mean_across_hospitals"] = df_s[f"{m}_mean"].mean()
            row[f"{m}_sd_across_hospitals"] = df_s[f"{m}_mean"].std()
        rows.append(row)
    return pd.DataFrame(rows)


def run_friedman_wilcoxon(df_agg, label, output_suffix):
    """Run Friedman and post-hoc Wilcoxon on hospital-level mean AUROC."""
    print("\n" + "=" * 80)
    print(f"STATISTICAL ANALYSIS -- {label}")
    print("=" * 80)

    pivot = df_agg.pivot_table(
        index="hospital",
        columns="strategy",
        values="auroc_mean",
    )
    strats_in = [s for s in STRATEGIES if s in pivot.columns]
    pivot_clean = pivot[strats_in].dropna()

    wilcoxon_rows = []
    if len(pivot_clean) >= 3 and len(strats_in) >= 3:
        groups = [pivot_clean[s].values for s in strats_in]
        chi2, p = stats.friedmanchisquare(*groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(
            f"\nFriedman ({len(strats_in)} strategies, n={len(pivot_clean)}): "
            f"chi2={chi2:.2f}, p={p:.6f} {sig}"
        )

        if p < 0.05:
            print("\nPost-hoc Wilcoxon (Bonferroni):")
            pairs = [
                (strats_in[i], strats_in[j])
                for i in range(len(strats_in))
                for j in range(i + 1, len(strats_in))
            ]
            n_comp = len(pairs)
            for sa, sb in pairs:
                va, vb = pivot_clean[sa].values, pivot_clean[sb].values
                try:
                    stat, p_w = stats.wilcoxon(va, vb)
                    p_bonf = min(p_w * n_comp, 1.0)
                    sig_str = "*" if p_bonf < 0.05 else "ns"
                    print(
                        f"  {sa:15s} vs {sb:15s}: "
                        f"median_diff={np.median(va - vb):+.4f}, "
                        f"p_bonf={p_bonf:.4f} {sig_str}"
                    )
                    wilcoxon_rows.append(
                        {
                            "analysis": label,
                            "strategy_a": sa,
                            "strategy_b": sb,
                            "median_diff": np.median(va - vb),
                            "p_wilcoxon": p_w,
                            "p_bonferroni": p_bonf,
                            "significant": p_bonf < 0.05,
                        }
                    )
                except ValueError:
                    pass
    else:
        print(
            f"\nNot enough complete paired observations for Friedman "
            f"(n={len(pivot_clean)}, strategies={len(strats_in)})."
        )

    if wilcoxon_rows:
        pd.DataFrame(wilcoxon_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"wilcoxon_tests_{output_suffix}.csv"),
            index=False,
        )

    best = pivot.idxmax(axis=1).value_counts()
    if len(best):
        print("\nBest strategy (AUROC, CV mean):")
        for s, c in best.items():
            print(f"  {s}: {c}/{len(pivot)} ({c / len(pivot) * 100:.0f}%)")


def print_final_summary(df_agg):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY -- REPEATED CV")
    print("=" * 80)

    for strat in STRATEGIES:
        df_s = df_agg[df_agg["strategy"] == strat]
        if len(df_s) == 0:
            continue
        print(f"\n  {strat} (n={len(df_s)} hospitals):")
        for m in ["auroc", "auprc", "brier_raw", "brier_cal"]:
            print(
                f"    {m:20s}: "
                f"{df_s[f'{m}_mean'].mean():.3f} +/- "
                f"{df_s[f'{m}_mean'].std():.3f}"
            )
        for m in ["f1_t05_cal", "f1_youden", "f1_optcal", "f2_optcal"]:
            print(
                f"    {m:20s}: "
                f"{df_s[f'{m}_mean'].mean():.3f} +/- "
                f"{df_s[f'{m}_mean'].std():.3f}"
            )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY BY VOLUME GROUP")
    print("=" * 80)
    for vol in sorted(df_agg[VOLUME_GROUP_COL].dropna().unique()):
        print(f"\n  {vol}")
        df_v = df_agg[df_agg[VOLUME_GROUP_COL] == vol]
        for strat in STRATEGIES:
            df_s = df_v[df_v["strategy"] == strat]
            if len(df_s) == 0:
                continue
            print(
                f"    {strat:15s} | n={len(df_s):2d} | "
                f"AUROC={df_s['auroc_mean'].mean():.3f}+/-{df_s['auroc_mean'].std():.3f} | "
                f"F1@You={df_s['f1_youden_mean'].mean():.3f} | "
                f"F2@opt={df_s['f2_optcal_mean'].mean():.3f}"
            )


# =============================================================================
# MAIN: REPEATED CV
# =============================================================================

def run_repeated_cv():
    """Main analysis: repeated 5-fold x 5 repetitions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    volume_csv_path = os.path.join(os.path.dirname(__file__), VOLUME_GROUPS_CSV)
    volume_df, volume_lookup, median_cutoff = load_hospital_volume_groups_csv(
        volume_csv_path
    )

    df_raw = pd.read_pickle(DATA_PATH)
    df = prepare_features(df_raw)
    df, hospital_volume_matches = attach_volume_groups(df, volume_lookup)
    hospital_volume_matches.to_csv(
        os.path.join(OUTPUT_DIR, "dataset_hospital_volume_matches.csv"),
        index=False,
    )
    validate_volume_assignments(df, hospital_volume_matches)

    df = df.dropna(subset=[VOLUME_GROUP_COL]).copy()
    feat_cols = get_feature_cols(df)

    print("=" * 80)
    print("PIPELINE v4 -- REPEATED 5-FOLD CV BY VOLUME GROUP")
    print("=" * 80)
    print(f"Patients: {len(df)} | Features: {len(feat_cols)}")
    print(f"Hospitals matched to volume groups: {df[HOSPITAL_COL].nunique()}")
    print(f"Mortality: {df[TARGET].mean():.3f}")
    print(
        f"CMBD median cutoff: {median_cutoff:.2f} mean valid discharges "
        f"(2020-2022)"
    )
    print(
        volume_df[VOLUME_GROUP_COL].value_counts()
        .rename_axis("volume_group")
        .to_string()
    )
    print(
        f"CV: {N_FOLDS}-fold x {N_REPEATS} repetitions = "
        f"{N_FOLDS * N_REPEATS} splits/hospital"
    )
    print(f"Calibration: {CAL_SIZE:.0%} of the strategy-specific train set")

    all_results = []
    hospital_info = []
    skipped = []

    hospitals = sorted(df[HOSPITAL_COL].dropna().unique())
    total_models = 0
    t_global = time.time()

    for i, hosp in enumerate(hospitals):
        hdata = df[df[HOSPITAL_COL] == hosp]
        n_total = len(hdata)
        n_pos = int(hdata[TARGET].sum())
        volume_group = hdata[VOLUME_GROUP_COL].iloc[0]

        if n_total == 0 or n_pos < N_FOLDS:
            skipped.append((hosp, f"n={n_total}, pos={n_pos} < {N_FOLDS} folds"))
            continue

        X_hosp = hdata[[c for c in feat_cols if c in hdata.columns]]
        y_hosp = hdata[TARGET]

        hospital_info.append(
            {
                "hospital": hosp,
                VOLUME_GROUP_COL: volume_group,
                "n_total": n_total,
                "n_pos": n_pos,
                "prevalence": n_pos / n_total,
            }
        )

        print(
            f"\n[{i + 1}/{len(hospitals)}] {hosp} ({volume_group}): "
            f"n={n_total}, mort={n_pos / n_total:.1%}"
        )

        hosp_strategy_aurocs = {s: [] for s in STRATEGIES}

        for rep in range(N_REPEATS):
            skf = StratifiedKFold(
                n_splits=N_FOLDS,
                shuffle=True,
                random_state=SEED + rep,
            )

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_hosp, y_hosp)):
                x_train_fold = X_hosp.iloc[train_idx]
                y_train_fold = y_hosp.iloc[train_idx]
                x_test_fold = X_hosp.iloc[test_idx]
                y_test_fold = y_hosp.iloc[test_idx]
                y_test_arr = y_test_fold.values

                if y_test_arr.sum() < MIN_POSITIVE_FOLD:
                    continue

                for strategy in STRATEGIES:
                    x_train, y_train = build_train_cv(
                        strategy,
                        df,
                        hosp,
                        x_train_fold,
                        y_train_fold,
                        feat_cols,
                    )

                    if x_train is None or len(x_train) == 0:
                        continue
                    if y_train.sum() == 0:
                        continue

                    model, platt, t_youden, t_f1, t_f2 = train_calibrate_within_fold(
                        x_train,
                        y_train,
                    )
                    y_proba_raw = model.predict_proba(x_test_fold)[:, 1]
                    metrics = evaluate_fold(
                        y_test_arr,
                        y_proba_raw,
                        platt,
                        t_youden,
                        t_f1,
                        t_f2,
                    )
                    if metrics is None:
                        continue

                    total_models += 1
                    all_results.append(
                        {
                            "hospital": hosp,
                            VOLUME_GROUP_COL: volume_group,
                            "strategy": strategy,
                            "repeat": rep,
                            "fold": fold_idx,
                            "n_train": len(x_train),
                            "n_pos_train": int(y_train.sum()),
                            **metrics,
                        }
                    )
                    hosp_strategy_aurocs[strategy].append(metrics["auroc"])

        for strat in STRATEGIES:
            vals = hosp_strategy_aurocs[strat]
            if vals:
                strat_rows = [
                    r for r in all_results
                    if r["hospital"] == hosp and r["strategy"] == strat
                ]
                print(
                    f"  {strat:15s} "
                    f"| AUROC={np.mean(vals):.3f}+/-{np.std(vals):.3f} "
                    f"| F1@.5={np.mean([r['f1_t05_cal'] for r in strat_rows]):.3f} "
                    f"| F1@You={np.mean([r['f1_youden'] for r in strat_rows]):.3f} "
                    f"| F1@opt={np.mean([r['f1_optcal'] for r in strat_rows]):.3f} "
                    f"| F2@You={np.mean([r['f2_youden'] for r in strat_rows]):.3f} "
                    f"| F2@opt={np.mean([r['f2_optcal'] for r in strat_rows]):.3f} "
                    f"| {len(vals)} folds"
                )

    df_cv = pd.DataFrame(all_results)
    df_hospitals = pd.DataFrame(hospital_info)

    elapsed = time.time() - t_global
    print(f"\n{'=' * 80}")
    print(f"CV COMPLETED: {total_models} models in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 80}")

    if skipped:
        pd.DataFrame(skipped, columns=["hospital", "reason"]).to_csv(
            os.path.join(OUTPUT_DIR, "skipped_hospitals.csv"),
            index=False,
        )

    df_agg = aggregate_cv_results(df_cv)
    df_strategy_summary = make_strategy_summary(df_agg)
    df_volume_strategy_summary = make_volume_strategy_summary(df_agg)

    run_friedman_wilcoxon(df_agg, "Global", "global")
    for vol in sorted(df_agg[VOLUME_GROUP_COL].dropna().unique()):
        run_friedman_wilcoxon(
            df_agg[df_agg[VOLUME_GROUP_COL] == vol],
            f"{vol}",
            vol.lower(),
        )

    print_final_summary(df_agg)

    df_cv.to_csv(os.path.join(OUTPUT_DIR, "cv_all_folds.csv"), index=False)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, "cv_aggregated.csv"), index=False)
    df_hospitals.to_csv(os.path.join(OUTPUT_DIR, "hospitals.csv"), index=False)
    df_strategy_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cv_summary_by_strategy.csv"),
        index=False,
    )
    df_volume_strategy_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cv_summary_by_volume_group.csv"),
        index=False,
    )

    print(f"\nSaved in {OUTPUT_DIR}/")
    return df_cv, df_agg, df_hospitals, volume_df


# =============================================================================
# SENSITIVITY: HOLDOUT
# =============================================================================

def run_holdout_sensitivity(df, feat_cols, df_agg_global):
    """Sensitivity analysis: one 75/25 holdout per hospital."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS -- HOLDOUT 75/25")
    print("=" * 80)

    hospitals = sorted(df[HOSPITAL_COL].dropna().unique())
    holdout_results = []

    for hosp in hospitals:
        hdata = df[df[HOSPITAL_COL] == hosp]
        if len(hdata) == 0 or hdata[TARGET].sum() < 2:
            continue

        X_all = hdata[[c for c in feat_cols if c in hdata.columns]]
        y_all = hdata[TARGET]

        try:
            x_tr, x_te, y_tr, y_te = train_test_split(
                X_all,
                y_all,
                test_size=HOLDOUT_TEST_SIZE,
                stratify=y_all,
                random_state=SEED,
            )
        except ValueError:
            continue

        if y_te.sum() < 3:
            continue

        y_te_arr = y_te.values
        volume_group = hdata[VOLUME_GROUP_COL].iloc[0]

        for strategy in STRATEGIES:
            x_train, y_train = build_train_cv(
                strategy,
                df,
                hosp,
                x_tr,
                y_tr,
                feat_cols,
            )

            if x_train is None or len(x_train) == 0 or y_train.sum() == 0:
                continue

            model, platt, t_youden, t_f1, t_f2 = train_calibrate_within_fold(
                x_train,
                y_train,
            )
            y_proba_raw = model.predict_proba(x_te)[:, 1]
            metrics = evaluate_fold(y_te_arr, y_proba_raw, platt, t_youden, t_f1, t_f2)

            if metrics is None:
                continue

            holdout_results.append(
                {
                    "hospital": hosp,
                    VOLUME_GROUP_COL: volume_group,
                    "strategy": strategy,
                    **metrics,
                }
            )

    df_holdout = pd.DataFrame(holdout_results)

    print("\nCV vs Holdout comparison (mean +/- SD across hospitals):")
    print(
        f"  {'Strategy':15s} | {'CV AUROC':>15s} | {'HO AUROC':>15s} | "
        f"{'diff':>7s} | {'CV F1@You':>12s} | {'HO F1@You':>12s} | "
        f"{'CV F2@opt':>12s} | {'HO F2@opt':>12s}"
    )
    print("  " + "-" * 115)

    for strat in STRATEGIES:
        cv_s = df_agg_global[df_agg_global["strategy"] == strat]
        ho_s = df_holdout[df_holdout["strategy"] == strat]
        if len(cv_s) > 0 and len(ho_s) > 0:
            cv_auroc = f"{cv_s['auroc_mean'].mean():.3f}+/-{cv_s['auroc_mean'].std():.3f}"
            ho_auroc = f"{ho_s['auroc'].mean():.3f}+/-{ho_s['auroc'].std():.3f}"
            diff = ho_s["auroc"].mean() - cv_s["auroc_mean"].mean()
            cv_f1y = f"{cv_s['f1_youden_mean'].mean():.3f}"
            ho_f1y = f"{ho_s['f1_youden'].mean():.3f}"
            cv_f2o = f"{cv_s['f2_optcal_mean'].mean():.3f}"
            ho_f2o = f"{ho_s['f2_optcal'].mean():.3f}"
            print(
                f"  {strat:15s} | {cv_auroc:>15s} | {ho_auroc:>15s} | "
                f"{diff:+7.3f} | {cv_f1y:>12s} | {ho_f1y:>12s} | "
                f"{cv_f2o:>12s} | {ho_f2o:>12s}"
            )

    df_holdout.to_csv(os.path.join(OUTPUT_DIR, "holdout_results.csv"), index=False)
    if len(df_holdout):
        df_holdout.groupby([VOLUME_GROUP_COL, "strategy"]).agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            f1_youden_mean=("f1_youden", "mean"),
            f2_optcal_mean=("f2_optcal", "mean"),
            n_hospitals=("hospital", "nunique"),
        ).reset_index().to_csv(
            os.path.join(OUTPUT_DIR, "holdout_summary_by_volume_group.csv"),
            index=False,
        )

    print("\nIf CV-Holdout differences are small (<0.01), conclusions are robust.")
    return df_holdout


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    t_start = time.time()

    df_cv, df_agg, df_hospitals, df_volume = run_repeated_cv()

    df_raw = pd.read_pickle(DATA_PATH)
    df = prepare_features(df_raw)
    volume_csv_path = os.path.join(os.path.dirname(__file__), VOLUME_GROUPS_CSV)
    _, volume_lookup, _ = load_hospital_volume_groups_csv(volume_csv_path)
    df, hospital_volume_matches = attach_volume_groups(df, volume_lookup)
    validate_volume_assignments(df, hospital_volume_matches)
    df = df.dropna(subset=[VOLUME_GROUP_COL]).copy()
    feat_cols = get_feature_cols(df)

    df_holdout = run_holdout_sensitivity(df, feat_cols, df_agg)

    total_time = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"FULL PIPELINE COMPLETED: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 80}")
