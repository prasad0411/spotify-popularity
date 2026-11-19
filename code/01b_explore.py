from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SONGS = ROOT / "data" / "songs_clean.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

HIT_THRESHOLD = 60

df = pd.read_csv(SONGS)

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "explicit", "time_signature",
]

# ---- Figure 1: popularity distribution ----
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["popularity"], bins=50, color="#4C72B0", edgecolor="white")
ax.axvline(HIT_THRESHOLD, color="#C44E52", linestyle="--",
           label="Hit cutoff (popularity = 60)")
ax.set_xlabel("Popularity score (0-100)")
ax.set_ylabel("Number of songs")
ax.set_title("Distribution of track popularity (one row per song)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig1_popularity_distribution.png", dpi=200)
plt.close(fig)

# ---- Figure 2: skew before and after log transform ----
pairs = [
    ("instrumentalness", "instrumentalness_log"),
    ("speechiness", "speechiness_log"),
    ("duration_ms", "duration_log"),
]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for j, (raw, logged) in enumerate(pairs):
    axes[0, j].hist(df[raw], bins=50, color="#4C72B0", edgecolor="white")
    axes[0, j].set_title("%s (raw)" % raw)
    axes[1, j].hist(df[logged], bins=50, color="#55A868", edgecolor="white")
    axes[1, j].set_title("%s (log scale)" % raw)
    axes[1, j].set_xlabel("Value after transform")
axes[0, 0].set_ylabel("Number of songs")
axes[1, 0].set_ylabel("Number of songs")
fig.suptitle("Skewed features before and after log transform")
fig.tight_layout()
fig.savefig(FIG / "fig2_feature_skew.png", dpi=200)
plt.close(fig)

# ---- Figure 3: correlation heatmap ----
cols = FEATURES + ["popularity"]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=60, ha="right")
ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, "%.2f" % corr.values[i, j], ha="center", va="center",
                fontsize=6.5)
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_title("Correlations among audio features and popularity")
fig.tight_layout()
fig.savefig(FIG / "fig3_correlation_heatmap.png", dpi=200)
plt.close(fig)

# ---- Numbers to check and quote in the report ----
print("Songs:", len(df))
print("\nCorrelation with popularity (Pearson):")
print(corr["popularity"].drop("popularity").sort_values(key=abs, ascending=False).round(3))
print("\nSkewness before/after log:")
for raw, logged in pairs:
    print("%s: %.2f -> %.2f" % (raw, df[raw].skew(), df[logged].skew()))
print("\nPopularity skewness: %.2f" % df["popularity"].skew())
print("Share of songs with popularity 0: %.3f" % (df["popularity"] == 0).mean())
print("\nFigures saved in:", FIG)