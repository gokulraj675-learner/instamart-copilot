"""
part_b.py -- Instamart Capstone Part B
========================================
1.  RFM computation (snapshot: 2025-10-01)
2.  K-Means clustering  K=2..8, forced final K=4
3.  PCA(n_components=2) visualisation coordinates
4.  Save  outputs/customer_segments.csv
5.  Lapse prediction  (cutoff: 2025-09-01)
      - Feature engineering on pre-cutoff orders only
      - Pipeline: OneHotEncoder(drop='first') + StandardScaler
      - GridSearchCV on LogisticRegression and RandomForestClassifier
      - Class-balance experiment (random over-sampling on train set)
6.  Save best model -> outputs/lapse_model.joblib
7.  Save Premium-segment retention list -> outputs/retention_list.csv

Allowed libraries: standard Python, pandas, numpy, scikit-learn, joblib, sqlite3
"""

import os
import sys
import sqlite3
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")          # suppress convergence / deprecation noise

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────
CLEAN        = os.path.join("data", "clean")
OUTPUTS      = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

RFM_SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
FINAL_K      = 4
RANDOM_STATE = 42

SEP  = "=" * 65
SEP2 = "-" * 65

# ─────────────────────────────────────────────────────────────
# 1.  LOAD CLEAN DATA
# ─────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1 - Loading clean data")
print(SEP)

cust    = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
orders  = pd.read_csv(os.path.join(CLEAN, "orders.csv"),
                      parse_dates=["order_ts"])
products = pd.read_csv(os.path.join("data", "raw", "products.csv"))

print(f"  customers : {cust.shape[0]:,} rows")
print(f"  orders    : {orders.shape[0]:,} rows  "
      f"[{orders['order_ts'].min().date()} .. {orders['order_ts'].max().date()}]")
print(f"  products  : {products.shape[0]:,} rows")

# ─────────────────────────────────────────────────────────────
# 2.  RFM COMPUTATION  (snapshot = 2025-10-01)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 - RFM computation  (snapshot: 2025-10-01)")
print(SEP)

# Use only completed orders for RFM (cancelled orders were never paid)
completed = orders[orders["status"] == "completed"].copy()

rfm = (
    completed
    .groupby("customer_id")
    .agg(
        last_order_ts   = ("order_ts", "max"),
        frequency       = ("order_id", "nunique"),
        monetary        = ("order_amount", "sum"),
    )
    .reset_index()
)
rfm["recency"] = (RFM_SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm.drop(columns=["last_order_ts"])

print(f"  Customers with at least 1 completed order : {len(rfm):,}")
print(f"  RFM stats:")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────
# 3.  K-MEANS CLUSTERING  (K=2..8, final K=4)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 - K-Means clustering")
print(SEP)

rfm_features = rfm[["recency", "frequency", "monetary"]].copy()

# Standardise
scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm_features)

# Elbow: inertia for K=2..8
print(f"\n  {'K':>3}   {'Inertia':>12}   {'Silhouette (approx)':>20}")
print(f"  {'-'*3}   {'-'*12}   {'-'*20}")

from sklearn.metrics import silhouette_score

inertias, silhouettes = {}, {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias[k]    = km.inertia_
    silhouettes[k] = silhouette_score(rfm_scaled, labels, sample_size=min(2000, len(rfm_scaled)))
    print(f"  {k:>3}   {inertias[k]:>12.1f}   {silhouettes[k]:>20.4f}")

# Final model: forced K=4
print(f"\n  Forcing final model: K = {FINAL_K}")
km_final = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=10)
rfm["segment_id"] = km_final.fit_predict(rfm_scaled)

# ─────────────────────────────────────────────────────────────
# 4.  PCA(n_components=2)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 - PCA (n_components=2)")
print(SEP)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(rfm_scaled)
rfm["pca_x"] = pca_coords[:, 0]
rfm["pca_y"] = pca_coords[:, 1]

print(f"  Explained variance ratio : PC1={pca.explained_variance_ratio_[0]:.3f}  "
      f"PC2={pca.explained_variance_ratio_[1]:.3f}  "
      f"Total={sum(pca.explained_variance_ratio_):.3f}")

# ─────────────────────────────────────────────────────────────
# 5.  LABEL SEGMENTS  (based on cluster RFM centroids)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 - Segment labelling")
print(SEP)

# Compute mean RFM per cluster in original (un-scaled) space
cluster_profiles = (
    rfm.groupby("segment_id")[["recency", "frequency", "monetary"]]
    .mean()
    .round(2)
)
print("\n  Cluster profiles (unscaled means):")
print(cluster_profiles.to_string())

# Rank clusters by composite score:
#   low recency = good, high frequency = good, high monetary = good
# Score = -recency_rank + frequency_rank + monetary_rank
cluster_profiles["_rec_rank"] = cluster_profiles["recency"].rank()          # low recency -> low rank -> bad sign
cluster_profiles["_freq_rank"] = cluster_profiles["frequency"].rank()
cluster_profiles["_mon_rank"]  = cluster_profiles["monetary"].rank()
cluster_profiles["_score"] = (
    -cluster_profiles["_rec_rank"]
    + cluster_profiles["_freq_rank"]
    + cluster_profiles["_mon_rank"]
)

score_order = cluster_profiles["_score"].sort_values(ascending=False)
label_map   = {
    int(score_order.index[0]): "Premium",
    int(score_order.index[1]): "Loyal",
    int(score_order.index[2]): "At-Risk",
    int(score_order.index[3]): "Lapsed",
}
rfm["segment_label"] = rfm["segment_id"].map(label_map)

print("\n  Segment label assignment:")
for sid, label in sorted(label_map.items()):
    n = (rfm["segment_id"] == sid).sum()
    print(f"    Cluster {sid} -> {label:<10s}  ({n:,} customers)")

seg_summary = (
    rfm.groupby("segment_label")[["recency", "frequency", "monetary"]]
    .agg(["mean", "count"])
    .round(2)
)
print("\n  Segment summary (mean | count):")
print(seg_summary.to_string())

# ─────────────────────────────────────────────────────────────
# 6.  SAVE customer_segments.csv
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 6 - Saving customer_segments.csv")
print(SEP)

segments_out = rfm[["customer_id", "recency", "frequency", "monetary",
                     "segment_id", "segment_label", "pca_x", "pca_y"]].copy()
# Merge customer metadata
segments_out = segments_out.merge(
    cust[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id", how="left"
)
seg_path = os.path.join(OUTPUTS, "customer_segments.csv")
segments_out.to_csv(seg_path, index=False)
print(f"  Saved -> {seg_path}  ({len(segments_out):,} rows)")
print(f"  Columns: {segments_out.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# 7.  LAPSE PREDICTION  (cutoff: 2025-09-01)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 7 - Lapse prediction feature engineering")
print(SEP)

# Definition:
#   lapsed = 1 if customer placed NO completed order on or after 2025-09-01
#   lapsed = 0 if customer placed at least one completed order >= 2025-09-01
#
# Features must be built ONLY from orders strictly BEFORE 2025-09-01.

pre  = orders[orders["order_ts"] < LAPSE_CUTOFF].copy()
post = orders[orders["order_ts"] >= LAPSE_CUTOFF].copy()

# Customers who had at least one order before cutoff (our modelling population)
active_before = set(pre["customer_id"].unique())
print(f"  Customers with >=1 order before cutoff : {len(active_before):,}")

# Customers with a completed order after cutoff = NOT lapsed
retained = set(
    post[post["status"] == "completed"]["customer_id"].unique()
)
print(f"  Customers retained (>=1 completed post-cutoff) : {len(retained):,}")

# Lapse label
lapse_df = pd.DataFrame({"customer_id": sorted(active_before)})
lapse_df["lapsed"] = lapse_df["customer_id"].apply(
    lambda c: 0 if c in retained else 1
)
print(f"  Lapsed (label=1)  : {lapse_df['lapsed'].sum():,}  "
      f"({100*lapse_df['lapsed'].mean():.1f}%)")
print(f"  Active (label=0)  : {(lapse_df['lapsed']==0).sum():,}")

# ─── Feature engineering from pre-cutoff orders ──────────────
print()
print(SEP2)
print("  Building features from pre-cutoff orders")
print(SEP2)

# Merge category onto pre-orders
pre = pre.merge(products[["product_id", "category"]], on="product_id", how="left")

def build_features(df_pre: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Compute per-customer features from a slice of orders."""
    grp = df_pre.groupby("customer_id")
    completed_pre = df_pre[df_pre["status"] == "completed"]
    cancelled_pre = df_pre[df_pre["status"] == "cancelled"]

    # Core RFM on pre-cutoff window
    rfm_pre = (
        completed_pre
        .groupby("customer_id")
        .agg(
            recency_days  = ("order_ts", lambda s: (snapshot - s.max()).days),
            frequency     = ("order_id", "nunique"),
            monetary_sum  = ("order_amount", "sum"),
            monetary_mean = ("order_amount", "mean"),
            monetary_max  = ("order_amount", "max"),
        )
    )

    # Order-level stats (all orders, not just completed)
    all_stats = (
        grp
        .agg(
            total_orders          = ("order_id", "nunique"),
            avg_discount_pct      = ("discount_pct", "mean"),
            max_discount_pct      = ("discount_pct", "max"),
            avg_quantity          = ("quantity", "mean"),
        )
    )

    # Cancel rate
    cancel_cnt = (
        cancelled_pre
        .groupby("customer_id")["order_id"]
        .nunique()
        .rename("cancel_count")
    )
    cancel_stats = cancel_cnt.to_frame().join(
        all_stats[["total_orders"]], how="right"
    ).fillna(0)
    cancel_stats["cancel_rate"] = (
        cancel_stats["cancel_count"] / cancel_stats["total_orders"]
    ).fillna(0)

    # Category diversity (number of distinct product categories ordered)
    cat_div = (
        df_pre.groupby("customer_id")["category"]
        .nunique()
        .rename("category_diversity")
    )

    # Inter-order gap (mean days between consecutive completed orders)
    def mean_gap(series):
        dates = sorted(series)
        if len(dates) < 2:
            return np.nan
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        return np.mean(gaps)

    inter_gap = (
        completed_pre
        .groupby("customer_id")["order_ts"]
        .apply(mean_gap)
        .rename("mean_inter_order_gap_days")
    )

    # Late delivery rate (delivery_minutes > 45)
    completed_with_deliv = completed_pre[completed_pre["delivery_minutes"].notna()].copy()
    late_rate = (
        completed_with_deliv
        .assign(is_late=completed_with_deliv["delivery_minutes"] > 45)
        .groupby("customer_id")["is_late"]
        .mean()
        .rename("late_delivery_rate")
    )

    # Preferred category (most ordered category by order count)
    pref_cat = (
        df_pre
        .groupby(["customer_id", "category"])["order_id"]
        .nunique()
        .reset_index()
        .sort_values("order_id", ascending=False)
        .drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
        .rename("preferred_category")
    )

    # Days since signup at cutoff
    signup_days = (
        cust.set_index("customer_id")["signup_date"]
        .apply(pd.Timestamp)
    )
    days_since_signup = (snapshot - signup_days).dt.days.rename("days_since_signup")

    # Assemble
    feat = (
        rfm_pre
        .join(all_stats, how="outer")
        .join(cancel_stats[["cancel_rate"]], how="left")
        .join(cat_div, how="left")
        .join(inter_gap, how="left")
        .join(late_rate, how="left")
        .join(pref_cat, how="left")
        .join(days_since_signup, how="left")
        .reset_index()
    )
    feat = feat.rename(columns={"index": "customer_id"})
    return feat

features = build_features(pre, LAPSE_CUTOFF)
print(f"  Feature table shape : {features.shape}")
print(f"  Feature columns : {features.columns.tolist()}")

# Merge features with labels (only customers who had pre-cutoff orders)
model_df = lapse_df.merge(features, on="customer_id", how="left")

# Customers with no completed order before cutoff have NaN rfm features
# Fill inter_order_gap and late_delivery_rate NaNs with 0
fill_zero_cols = ["mean_inter_order_gap_days", "late_delivery_rate"]
for col in fill_zero_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(0)

# Fill remaining NaN numerics with 0 (customers with only cancelled orders)
numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["lapsed"]]
model_df[numeric_cols] = model_df[numeric_cols].fillna(0)

print(f"  Model dataset shape : {model_df.shape}")
print(f"  Missing values per column:\n"
      f"{model_df.isnull().sum()[model_df.isnull().sum() > 0].to_string()}")

# ─────────────────────────────────────────────────────────────
# 8.  TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 8 - Train / test split (80/20 stratified)")
print(SEP)

FEATURE_COLS  = [c for c in model_df.columns
                 if c not in ("customer_id", "lapsed")]
CATEGORICAL   = ["preferred_category"]
NUMERICAL     = [c for c in FEATURE_COLS if c not in CATEGORICAL]

X = model_df[FEATURE_COLS].copy()
y = model_df["lapsed"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"  Train : {len(X_train):,} samples  (lapse rate {y_train.mean():.2%})")
print(f"  Test  : {len(X_test):,} samples   (lapse rate {y_test.mean():.2%})")

# ─────────────────────────────────────────────────────────────
# 9.  PIPELINE  (fitted ONLY on train set)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 9 - Pipeline: OneHotEncoder + StandardScaler")
print(SEP)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                               sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERICAL),
    ],
    remainder="drop",
)

# ─────────────────────────────────────────────────────────────
# 10. GRIDSEARCHCV - LogisticRegression
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 10 - GridSearchCV: LogisticRegression")
print(SEP)

lr_pipe = Pipeline([
    ("prep",  preprocessor),
    ("clf",   LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
    )),
])

lr_param_grid = {
    "clf__C":       [0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l2"],
    "clf__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lr_gs = GridSearchCV(
    lr_pipe, lr_param_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, verbose=0, refit=True
)
lr_gs.fit(X_train, y_train)

lr_best = lr_gs.best_estimator_
y_pred_lr  = lr_best.predict(X_test)
y_prob_lr  = lr_best.predict_proba(X_test)[:, 1]

print(f"  Best params : {lr_gs.best_params_}")
print(f"  CV ROC-AUC  : {lr_gs.best_score_:.4f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"  Test F1     : {f1_score(y_test, y_pred_lr):.4f}")
print(f"\n  Classification Report (LR):")
print(classification_report(y_test, y_pred_lr, target_names=["Active","Lapsed"]))

# ─────────────────────────────────────────────────────────────
# 11. GRIDSEARCHCV - RandomForestClassifier
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 11 - GridSearchCV: RandomForestClassifier")
print(SEP)

rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf",  RandomForestClassifier(random_state=RANDOM_STATE)),
])

rf_param_grid = {
    "clf__n_estimators":      [100, 200],
    "clf__max_depth":         [None, 10, 20],
    "clf__min_samples_leaf":  [1, 5],
    "clf__class_weight":      [None, "balanced"],
}

rf_gs = GridSearchCV(
    rf_pipe, rf_param_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, verbose=0, refit=True
)
rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_prob_rf = rf_best.predict_proba(X_test)[:, 1]

print(f"  Best params : {rf_gs.best_params_}")
print(f"  CV ROC-AUC  : {rf_gs.best_score_:.4f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  Test F1     : {f1_score(y_test, y_pred_rf):.4f}")
print(f"\n  Classification Report (RF):")
print(classification_report(y_test, y_pred_rf, target_names=["Active","Lapsed"]))

# Feature importance (RF)
try:
    feature_names_out = (
        rf_best.named_steps["prep"]
        .get_feature_names_out()
        .tolist()
    )
    importances = rf_best.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=feature_names_out).sort_values(ascending=False)
    print("\n  Top-10 feature importances (RF):")
    print(fi.head(10).round(4).to_string())
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# 12. CLASS-BALANCE EXPERIMENT (random over-sampling on train)
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 12 - Class-balance experiment (random over-sampling)")
print(SEP)

def random_oversample(X_df: pd.DataFrame, y_arr: np.ndarray,
                      random_state: int = RANDOM_STATE):
    """Upsample the minority class to match the majority class size."""
    rng    = np.random.RandomState(random_state)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    majority_idx = neg_idx if len(neg_idx) > len(pos_idx) else pos_idx
    minority_idx = pos_idx if len(neg_idx) > len(pos_idx) else neg_idx
    n_upsample = len(majority_idx) - len(minority_idx)
    extra_idx = rng.choice(minority_idx, size=n_upsample, replace=True)
    all_idx = np.concatenate([np.arange(len(y_arr)), extra_idx])
    rng.shuffle(all_idx)
    return X_df.iloc[all_idx].reset_index(drop=True), y_arr[all_idx]

X_train_os, y_train_os = random_oversample(X_train, y_train)
print(f"  Before over-sampling : {len(y_train):,}  "
      f"(lapse={y_train.sum():,}  active={(y_train==0).sum():,})")
print(f"  After  over-sampling : {len(y_train_os):,}  "
      f"(lapse={y_train_os.sum():,}  active={(y_train_os==0).sum():,})")

# Refit LR on over-sampled set (best params, no class_weight since we balance manually)
lr_os_params = {k.replace("clf__",""):v for k,v in lr_gs.best_params_.items()}
lr_os_params.pop("class_weight", None)   # balance handled by over-sampling
lr_os = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE,
        **{k:v for k,v in lr_os_params.items() if k in ("C","penalty")}
    )),
])
lr_os.fit(X_train_os, y_train_os)
y_pred_lr_os = lr_os.predict(X_test)
y_prob_lr_os = lr_os.predict_proba(X_test)[:, 1]

# Refit RF on over-sampled set
rf_os_params = {k.replace("clf__",""):v for k,v in rf_gs.best_params_.items()}
rf_os_params.pop("class_weight", None)
rf_os = Pipeline([
    ("prep", preprocessor),
    ("clf",  RandomForestClassifier(
        random_state=RANDOM_STATE,
        **{k:v for k,v in rf_os_params.items() if k in ("n_estimators","max_depth","min_samples_leaf")}
    )),
])
rf_os.fit(X_train_os, y_train_os)
y_pred_rf_os = rf_os.predict(X_test)
y_prob_rf_os = rf_os.predict_proba(X_test)[:, 1]

print("\n  Results comparison on TEST set:")
print(f"  {'Model':<35} {'ROC-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

results = {
    "LR  (original train)"    : (y_prob_lr,    y_pred_lr),
    "LR  (over-sampled)"      : (y_prob_lr_os, y_pred_lr_os),
    "RF  (original train)"    : (y_prob_rf,    y_pred_rf),
    "RF  (over-sampled)"      : (y_prob_rf_os, y_pred_rf_os),
}
best_model_name = None
best_auc        = -1.0
best_pipeline   = None

for name, (proba, pred) in results.items():
    auc   = roc_auc_score(y_test, proba)
    f1    = f1_score(y_test, pred)
    prec  = precision_score(y_test, pred, zero_division=0)
    rec   = recall_score(y_test, pred, zero_division=0)
    print(f"  {name:<35} {auc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")
    if auc > best_auc:
        best_auc        = auc
        best_model_name = name
        # Map back to pipeline objects
        if name.startswith("LR"):
            best_pipeline = lr_os if "over" in name else lr_best
        else:
            best_pipeline = rf_os if "over" in name else rf_best

print(f"\n  Best model : {best_model_name}  (ROC-AUC = {best_auc:.4f})")

# ─────────────────────────────────────────────────────────────
# 13. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 13 - Saving best model")
print(SEP)

model_path = os.path.join(OUTPUTS, "lapse_model.joblib")
joblib.dump(best_pipeline, model_path, compress=3)
print(f"  Saved -> {model_path}")
print(f"  Model type : {type(best_pipeline.named_steps['clf']).__name__}")

# Verify round-trip load
loaded_model = joblib.load(model_path)
y_prob_verify = loaded_model.predict_proba(X_test)[:, 1]
print(f"  Round-trip ROC-AUC : {roc_auc_score(y_test, y_prob_verify):.4f}  [should match {best_auc:.4f}]")

# ─────────────────────────────────────────────────────────────
# 14. PREMIUM SEGMENT RETENTION LIST
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 14 - Premium segment retention list")
print(SEP)

# Get Premium customer_ids from segmentation
premium_ids = set(
    segments_out[segments_out["segment_label"] == "Premium"]["customer_id"]
)
print(f"  Total Premium segment customers : {len(premium_ids):,}")

# Get lapse probability for every customer in the model dataset
model_df["lapse_prob"] = best_pipeline.predict_proba(X[FEATURE_COLS])[:, 1]

# Retention list = Premium customers who have a lapse probability > 0.4
# (prioritise customers at real risk of churning)
premium_model = model_df[model_df["customer_id"].isin(premium_ids)].copy()

retention = (
    premium_model
    .merge(
        # Only pull city/age/gender/segment_label from segments_out to avoid
        # column name collisions with recency/frequency/monetary in model_df
        segments_out[["customer_id", "city", "age", "gender", "segment_label"]],
        on="customer_id", how="left"
    )
    .sort_values("lapse_prob", ascending=False)
    [["customer_id", "city", "age", "gender",
      "recency_days", "frequency", "monetary_sum",
      "lapsed", "lapse_prob", "segment_label"]]
    .rename(columns={
        "recency_days": "recency",
        "frequency":    "frequency_pre_cutoff",
        "monetary_sum": "monetary_pre_cutoff",
    })
)

retention_path = os.path.join(OUTPUTS, "retention_list.csv")
retention.to_csv(retention_path, index=False)
print(f"  Saved -> {retention_path}  ({len(retention):,} rows)")
print(f"  Customers at high lapse risk (prob > 0.5) : "
      f"{(retention['lapse_prob'] > 0.5).sum():,}")
print(f"  Avg lapse probability (Premium segment) : "
      f"{retention['lapse_prob'].mean():.3f}")
print()
print(retention[["customer_id","city","recency","frequency_pre_cutoff","monetary_pre_cutoff","lapse_prob","lapsed"]]
      .head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 15. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART B COMPLETE")
print(SEP)
print()
print("Outputs written:")
print(f"  {os.path.join(OUTPUTS, 'customer_segments.csv')}  ({len(segments_out):,} rows)")
print(f"  {model_path}")
print(f"  {retention_path}  ({len(retention):,} rows)")
print()
print("Segment distribution:")
print(segments_out["segment_label"].value_counts().to_string())
print()
print(f"Best lapse model : {best_model_name}  |  Test ROC-AUC = {best_auc:.4f}")
