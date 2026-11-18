from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "dataset.csv"
OUT_SONGS = ROOT / "data" / "songs_clean.csv"
OUT_GENRE = ROOT / "data" / "songs_by_genre_clean.csv"

HIT_THRESHOLD = 60  # team decision: hit = popularity >= 60

df = pd.read_csv(RAW)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("Rows in raw file:", len(df))
print("Unique track_id:", df["track_id"].nunique())
print("Genres:", df["track_genre"].nunique())
print("Missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])
print("Rows with duration <= 0:", (df["duration_ms"] <= 0).sum())

# 1. Drop rows with missing text and the zero-duration row
df = df.dropna(subset=["artists", "album_name", "track_name"])
df = df[df["duration_ms"] > 0]
df["explicit"] = df["explicit"].astype(int)

# 2. Song-by-genre table: one row per song per genre
by_genre = df.drop_duplicates(subset=["track_id", "track_genre"]).copy()

# 3. One-row-per-song table
pop_spread = df.groupby("track_id")["popularity"].nunique()
print("Songs whose popularity differs between genre rows:", (pop_spread > 1).sum())

songs = df.sort_values(["track_id", "popularity"], ascending=[True, False])
songs = songs.drop_duplicates(subset="track_id", keep="first").copy()
songs["n_genres"] = songs["track_id"].map(by_genre.groupby("track_id").size())
songs = songs.drop(columns=["track_genre"])

# 4. Log transforms for skewed features (raw columns are kept)
for table in (songs, by_genre):
    table["instrumentalness_log"] = np.log1p(table["instrumentalness"])
    table["speechiness_log"] = np.log1p(table["speechiness"])
    table["duration_log"] = np.log(table["duration_ms"])
    table["is_hit"] = (table["popularity"] >= HIT_THRESHOLD).astype(int)

# 5. Report the numbers the team needs
print("\n--- Cleaned data ---")
print("Unique songs:", len(songs))
print("Song-by-genre rows:", len(by_genre))
print("Mean popularity (songs):", round(songs["popularity"].mean(), 2))
print("90th percentile of popularity (songs):", songs["popularity"].quantile(0.90))
print("Share of hits at >= %d:" % HIT_THRESHOLD, round(songs["is_hit"].mean(), 4))

# 6. Save
songs.to_csv(OUT_SONGS, index=False)
by_genre.to_csv(OUT_GENRE, index=False)
print("\nSaved:", OUT_SONGS.name, "and", OUT_GENRE.name)