# What Makes a Song Popular?

Predicting and explaining Spotify track popularity from audio features.

## Team
Tejashree Venkatesh, Chaitanya Sandeep Jumale, Prasad Chandrashekhar Kanade

## Abstract
We study how well a track's intrinsic audio features explain and predict its
popularity on Spotify, and whether that relationship holds across genres. Using
the public Spotify Tracks Dataset (about 114,000 rows covering roughly 89,700
unique tracks across 114 genres), we fit regression, classification, and clustering models, evaluate them against
naive baselines, and test whether audio to popularity patterns differ by genre.

## Dataset
Not stored in this repo. Download dataset.csv into the data/ folder from:
https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

## Structure
- data/    dataset.csv goes here (gitignored)
- code/    analysis scripts (01_clean, 02_models, 03_genre_tests)
- figures/ exported plots

## How to run
1. Place dataset.csv in data/
2. Run scripts in code/ in numeric order
