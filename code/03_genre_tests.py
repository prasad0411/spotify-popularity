# Problem 3: genre analysis, hypothesis tests, and clustering.
# Reads the cleaned files Tejashree produced, saves figures to figures/.
# Run from the repo root: python3 code/03_genre_tests.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

DATA = "data"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

songs = pd.read_csv(os.path.join(DATA, "songs_clean.csv"))
gen = pd.read_csv(os.path.join(DATA, "songs_by_genre_clean.csv"))

# audio features (log versions for the skewed ones, matching cleaning)
feats = ["danceability", "energy", "loudness", "speechiness_log", "acousticness",
         "instrumentalness_log", "liveness", "valence", "tempo", "duration_log"]

# ---------- 1. Is the audio to popularity relationship genre dependent? ----------
# Fit one linear model per genre and compare the spread of R squared values
# against a single pooled model that ignores genre.
pooled_r2 = LinearRegression().fit(gen[feats], gen.popularity).score(gen[feats], gen.popularity)

r2_rows = []
for genre, grp in gen.groupby("track_genre"):
    if len(grp) >= 100:
        r2 = LinearRegression().fit(grp[feats], grp.popularity).score(grp[feats], grp.popularity)
        r2_rows.append((genre, r2))
r2df = pd.DataFrame(r2_rows, columns=["genre", "r2"]).sort_values("r2", ascending=False)

print("Pooled audio-only R2:", round(pooled_r2, 3))
print("Per-genre R2: median %.3f, min %.3f, max %.3f" %
      (r2df.r2.median(), r2df.r2.min(), r2df.r2.max()))

plt.figure(figsize=(8, 5))
plt.hist(r2df.r2, bins=25, color="steelblue", edgecolor="white")
plt.axvline(pooled_r2, color="red", linewidth=2,
            label="Pooled model R2 = %.3f" % pooled_r2)
plt.axvline(r2df.r2.median(), color="black", linestyle="--",
            label="Per-genre median = %.3f" % r2df.r2.median())
plt.title("Audio features explain popularity far better within some genres")
plt.xlabel("R squared of a linear model fit within one genre")
plt.ylabel("Number of genres")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_genre_r2.png"), dpi=150)
plt.close()

# ---------- 2. Hypothesis test: do explicit songs differ in popularity? ----------
# Overall Welch t-test on one row per song, with Cohen's d and a 95% CI.
a = songs.loc[songs.explicit == 1, "popularity"]
b = songs.loc[songs.explicit == 0, "popularity"]
t, p = stats.ttest_ind(a, b, equal_var=False)
nx, ny = len(a), len(b)
sp = np.sqrt(((nx - 1) * a.var() + (ny - 1) * b.var()) / (nx + ny - 2))
d = (a.mean() - b.mean()) / sp
se_d = np.sqrt((nx + ny) / (nx * ny) + d ** 2 / (2 * (nx + ny)))
print("\nExplicit vs non-explicit (overall): explicit %.2f, non-explicit %.2f" %
      (a.mean(), b.mean()))
print("Welch t=%.2f, p=%.2e, Cohen d=%.3f, 95%% CI [%.3f, %.3f]" %
      (t, p, d, d - 1.96 * se_d, d + 1.96 * se_d))

# Per-genre version with Benjamini-Hochberg control for multiple testing.
def bh_reject(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * np.arange(1, n + 1) / n
    below = ranked <= thresh
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    rej = np.zeros(n, bool)
    if k > 0:
        rej[order[:k]] = True
    return rej

rows = []
for genre, grp in gen.groupby("track_genre"):
    e = grp.loc[grp.explicit == 1, "popularity"]
    n_ = grp.loc[grp.explicit == 0, "popularity"]
    if len(e) >= 10 and len(n_) >= 10:
        tt, pp = stats.ttest_ind(e, n_, equal_var=False)
        rows.append((genre, e.mean() - n_.mean(), pp))
ex = pd.DataFrame(rows, columns=["genre", "mean_diff", "p"])
ex["sig"] = bh_reject(ex.p.values)
print("\nPer-genre explicit test: %d genres tested, %d significant after BH (%d higher, %d lower)" %
      (len(ex), ex.sig.sum(), (ex[ex.sig].mean_diff > 0).sum(), (ex[ex.sig].mean_diff < 0).sum()))

plt.figure(figsize=(8, 5))
plt.hist(ex.loc[~ex.sig, "mean_diff"], bins=20, color="lightgray",
         edgecolor="white", label="not significant")
plt.hist(ex.loc[ex.sig, "mean_diff"], bins=20, color="steelblue",
         edgecolor="white", label="significant (BH, 0.05)")
plt.axvline(0, color="red", linewidth=1)
plt.title("Explicit minus non-explicit mean popularity, by genre")
plt.xlabel("Mean popularity difference (explicit minus non-explicit)")
plt.ylabel("Number of genres")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_explicit_by_genre.png"), dpi=150)
plt.close()

# ---------- 3. Clustering: do audio features recover genre labels? ----------
Xg = StandardScaler().fit_transform(gen[feats])
km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(Xg)
ari = adjusted_rand_score(gen.track_genre, km.labels_)
print("\nK-means (k=10) vs 114 genres: adjusted Rand index = %.3f" % ari)

coords = PCA(n_components=2, random_state=42).fit_transform(Xg)
plt.figure(figsize=(8, 6))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=km.labels_, cmap="tab10",
                 s=3, alpha=0.3)
plt.title("Audio feature space (PCA) colored by k-means cluster, ARI = %.3f" % ari)
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar(sc, label="cluster")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_cluster_pca.png"), dpi=150)
plt.close()

print("\nSaved figures: fig_genre_r2.png, fig_explicit_by_genre.png, fig_cluster_pca.png")
