# Problem 2: Prediction models and hit classification.
# Reads songs_clean.csv from Tejashree's cleaning step.
# Run from the repo root: python3 code/02_models.py
# Author: Chaitanya Jumale

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    classification_report, RocCurveDisplay,
)

DATA = "data"
FIG  = "figures"
os.makedirs(FIG, exist_ok=True)
HIT_THRESHOLD = 60

# ─── load ────────────────────────────────────────────────────────────
songs = pd.read_csv(os.path.join(DATA, "songs_clean.csv"))
print(f"Loaded {len(songs)} songs, {songs.is_hit.sum()} hits ({songs.is_hit.mean():.1%})")

FEATURES = [
    "danceability", "energy", "loudness",
    "speechiness_log", "acousticness",
    "instrumentalness_log", "liveness", "valence",
    "tempo", "duration_log",
    "explicit", "key", "mode", "time_signature",
]

X = songs[FEATURES].values
y_reg = songs["popularity"].values
y_clf = songs["is_hit"].values
groups = songs["track_id"].values

# ─── 80/20 split by track_id (first fold of 5-fold GroupKFold) ──────
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X, y_reg, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]
y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]
groups_train = groups[train_idx]

# standardize (fit on train only — per Tejashree's note)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ─── 5-fold grouped CV helper ───────────────────────────────────────
cv_inner = GroupKFold(n_splits=5)

def cv_reg(name, model):
    rmses, r2s = [], []
    for tr, va in cv_inner.split(X_train_s, y_reg_train, groups_train):
        m = model.__class__(**model.get_params())
        m.fit(X_train_s[tr], y_reg_train[tr])
        p = m.predict(X_train_s[va])
        rmses.append(np.sqrt(mean_squared_error(y_reg_train[va], p)))
        r2s.append(r2_score(y_reg_train[va], p))
    model.fit(X_train_s, y_reg_train)
    pt = model.predict(X_test_s)
    cv_rmse, cv_r2 = np.mean(rmses), np.mean(r2s)
    t_rmse = np.sqrt(mean_squared_error(y_reg_test, pt))
    t_r2   = r2_score(y_reg_test, pt)
    print(f"  {name:28s}  CV-RMSE={cv_rmse:.3f}  CV-R2={cv_r2:.4f}  |  Test-RMSE={t_rmse:.3f}  Test-R2={t_r2:.4f}")
    return {"CV_RMSE": cv_rmse, "CV_R2": cv_r2, "Test_RMSE": t_rmse, "Test_R2": t_r2, "pred": pt}

def cv_clf(name, model):
    accs, f1s, aucs = [], [], []
    for tr, va in cv_inner.split(X_train_s, y_clf_train, groups_train):
        m = model.__class__(**model.get_params())
        m.fit(X_train_s[tr], y_clf_train[tr])
        p = m.predict(X_train_s[va])
        pr = m.predict_proba(X_train_s[va])[:, 1] if hasattr(m, "predict_proba") else np.zeros(len(p))
        accs.append(accuracy_score(y_clf_train[va], p))
        f1s.append(f1_score(y_clf_train[va], p, zero_division=0))
        aucs.append(roc_auc_score(y_clf_train[va], pr) if pr.any() else 0.5)
    model.fit(X_train_s, y_clf_train)
    pt = model.predict(X_test_s)
    prt = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(pt))
    cv_acc, cv_f1, cv_auc = np.mean(accs), np.mean(f1s), np.mean(aucs)
    t_acc = accuracy_score(y_clf_test, pt)
    t_f1  = f1_score(y_clf_test, pt, zero_division=0)
    t_auc = roc_auc_score(y_clf_test, prt) if prt.any() else 0.5
    print(f"  {name:28s}  CV-AUC={cv_auc:.4f}  CV-F1={cv_f1:.4f}  |  Test-AUC={t_auc:.4f}  Test-F1={t_f1:.4f}")
    return {"CV_Acc": cv_acc, "CV_F1": cv_f1, "CV_AUC": cv_auc,
            "Test_Acc": t_acc, "Test_F1": t_f1, "Test_AUC": t_auc, "pred": pt, "prob": prt}


# =====================================================================
#  REGRESSION
# =====================================================================
print("\n" + "="*60)
print("REGRESSION: predict popularity (continuous)")
print("="*60)

reg_specs = [
    ("Mean Baseline",     DummyRegressor(strategy="mean")),
    ("Linear Regression", LinearRegression()),
    ("Ridge (a=1.0)",     Ridge(alpha=1.0)),
    ("Lasso (a=0.1)",     Lasso(alpha=0.1)),
    ("Random Forest",     RandomForestRegressor(n_estimators=50, max_depth=10,
                                                min_samples_leaf=30, random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                    learning_rate=0.1, subsample=0.8, random_state=42)),
]

reg_results = {}
for name, model in reg_specs:
    t0 = time.time()
    reg_results[name] = cv_reg(name, model)
    print(f"    ({time.time()-t0:.1f}s)")

# pick best non-baseline
best_reg_name = max([k for k in reg_results if k != "Mean Baseline"],
                    key=lambda k: reg_results[k]["Test_R2"])

# =====================================================================
#  CLASSIFICATION
# =====================================================================
print("\n" + "="*60)
print("CLASSIFICATION: hit (popularity >= 60) vs non-hit")
print("="*60)

clf_specs = [
    ("Majority Baseline", DummyClassifier(strategy="most_frequent")),
    ("Random Forest",     RandomForestClassifier(n_estimators=50, max_depth=10,
                                                 min_samples_leaf=30,
                                                 class_weight="balanced",
                                                 random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                     learning_rate=0.1,
                                                     subsample=0.8, random_state=42)),
]

clf_results = {}
for name, model in clf_specs:
    t0 = time.time()
    clf_results[name] = cv_clf(name, model)
    print(f"    ({time.time()-t0:.1f}s)")

best_clf_name = max([k for k in clf_results if k != "Majority Baseline"],
                    key=lambda k: clf_results[k]["Test_AUC"])

print(f"\nBest classifier: {best_clf_name}")
print(classification_report(y_clf_test, clf_results[best_clf_name]["pred"],
                            target_names=["Non-hit", "Hit"]))

# refit best models to extract feature importances
for name, model in reg_specs:
    if name == best_reg_name:
        model.fit(X_train_s, y_reg_train)
        best_reg_model = model
for name, model in clf_specs:
    if name == best_clf_name:
        model.fit(X_train_s, y_clf_train)
        best_clf_model = model


# =====================================================================
#  FIGURES
# =====================================================================

# Fig 4: Regression comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names_r = list(reg_results.keys())
rmses = [reg_results[n]["Test_RMSE"] for n in names_r]
r2s   = [reg_results[n]["Test_R2"]   for n in names_r]
colors = ["#d9534f" if "Baseline" in n else "#5bc0de" for n in names_r]
axes[0].barh(names_r, rmses, color=colors, edgecolor="white")
axes[0].set_xlabel("Test RMSE (lower is better)")
axes[0].set_title("Regression: RMSE by model")
axes[0].invert_yaxis()
axes[1].barh(names_r, r2s, color=colors, edgecolor="white")
axes[1].set_xlabel("Test R² (higher is better)")
axes[1].set_title("Regression: R² by model")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_regression_comparison.png"), dpi=150)
plt.close()
print("Saved fig4_regression_comparison.png")

# Fig 5: Feature importance (regression)
if hasattr(best_reg_model, "feature_importances_"):
    imp = best_reg_model.feature_importances_; imp_label = "Gini importance"
elif hasattr(best_reg_model, "coef_"):
    imp = np.abs(best_reg_model.coef_); imp_label = "|coefficient|"
else:
    imp = np.zeros(len(FEATURES)); imp_label = "N/A"
imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values("importance", ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"], imp_df["importance"], color="steelblue", edgecolor="white")
plt.xlabel(f"Feature importance ({imp_label})")
plt.title(f"Feature importance — {best_reg_name} (regression)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_feature_importance_regression.png"), dpi=150)
plt.close()
print("Saved fig5_feature_importance_regression.png")

# Fig 6: Feature importance (classifier)
if hasattr(best_clf_model, "feature_importances_"):
    imp_c = best_clf_model.feature_importances_; imp_c_label = "Gini importance"
elif hasattr(best_clf_model, "coef_"):
    imp_c = np.abs(best_clf_model.coef_[0]); imp_c_label = "|coefficient|"
else:
    imp_c = np.zeros(len(FEATURES)); imp_c_label = "N/A"
imp_c_df = pd.DataFrame({"feature": FEATURES, "importance": imp_c}).sort_values("importance", ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(imp_c_df["feature"], imp_c_df["importance"], color="darkorange", edgecolor="white")
plt.xlabel(f"Feature importance ({imp_c_label})")
plt.title(f"Feature importance — {best_clf_name} (hit classifier)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_feature_importance_classifier.png"), dpi=150)
plt.close()
print("Saved fig6_feature_importance_classifier.png")

# Fig 7: ROC curve
fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_clf_test, clf_results[best_clf_name]["prob"],
                                 ax=ax, name=best_clf_name)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")
ax.set_title("ROC curve — hit vs non-hit classifier")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig7_roc_curve.png"), dpi=150)
plt.close()
print("Saved fig7_roc_curve.png")

# Fig 8: Actual vs predicted scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_reg_test, reg_results[best_reg_name]["pred"], s=3, alpha=0.15, color="steelblue")
ax.plot([0, 100], [0, 100], "r--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual popularity")
ax.set_ylabel("Predicted popularity")
ax.set_title(f"Actual vs Predicted — {best_reg_name}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig8_actual_vs_predicted.png"), dpi=150)
plt.close()
print("Saved fig8_actual_vs_predicted.png")

# ─── Summary ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY TABLES FOR REPORT")
print("="*60)
print("\nRegression:")
for n in reg_results:
    r = reg_results[n]
    print(f"  {n:28s}  CV-RMSE={r['CV_RMSE']:.3f}  CV-R2={r['CV_R2']:.4f}  Test-RMSE={r['Test_RMSE']:.3f}  Test-R2={r['Test_R2']:.4f}")
print("\nClassification:")
for n in clf_results:
    r = clf_results[n]
    print(f"  {n:28s}  CV-F1={r['CV_F1']:.4f}  CV-AUC={r['CV_AUC']:.4f}  Test-F1={r['Test_F1']:.4f}  Test-AUC={r['Test_AUC']:.4f}  Test-Acc={r['Test_Acc']:.4f}")

print("\nDone.")
