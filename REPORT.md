# What Makes a Song Popular? Predicting and Explaining Spotify Track Popularity From Audio Features

**Authors:** Tejashree Venkatesh, Chaitanya Sandeep Jumale, Prasad Chandrashekhar Kanade

**GitHub:** https://github.com/prasad0411/spotify-popularity

## Abstract

Streaming platforms shape how most music is discovered, yet it is unclear whether a song's intrinsic sound predicts how popular it becomes. Using the public Spotify Tracks Dataset of about 114,000 tracks across 114 genres, we study how well audio features explain and predict track popularity, whether that relationship depends on genre, and whether the features alone recover the genre labels. We fit regression and tree based models for popularity, a classifier for hit versus non hit (defined as the top ten percent of popularity), per genre models, hypothesis tests, and k means clustering, each evaluated against a naive baseline. Audio features carry only weak signal for popularity overall, the best model reaching an R squared of 0.13, but their explanatory power varies sharply by genre, and the audio space does not separate genres cleanly. We conclude that popularity is largely driven by factors outside the audio itself, while the features still capture meaningful, genre specific structure.

## Introduction

Music streaming has turned song discovery into an algorithmic process, where popularity scores influence playlisting, recommendation, and ultimately an artist's reach. For artists, labels, and curators, a natural question is whether something intrinsic to the sound of a track, how danceable, energetic, acoustic, or instrumental it is, can explain or anticipate its popularity. If it could, those signals would be available before a song accumulates any plays, independent of marketing or an artist's existing fame.

The domain problem is therefore to understand the relationship between a track's measurable audio qualities and its popularity. We translate this into three analytic problems: first, can audio features predict a popularity score and distinguish hits from non hits; second, does any such relationship hold uniformly across genres or vary between them; and third, do the audio features on their own correspond to the genre labels assigned to tracks. Together these let us judge both how much signal the audio carries and where that signal lives.

## Methods

**Data.** The Spotify Tracks Dataset is a single tabular file of 114,000 rows gathered through the Spotify Web API. Each row is a track in a given genre and carries a popularity score from 0 to 100, audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo), and metadata (genre, explicit flag, key, mode, time signature, duration). Because the same track is listed once per genre it belongs to, the file contains only 89,740 unique songs across 114 genres.

**Cleaning.** We removed one row that had missing text fields and a zero duration, and 450 exact repeats of a track within the same genre, and cast the explicit flag to an integer. We produced two tables: a song by genre table of 113,549 rows (one row per track per genre) for the genre analysis, and a one row per song table of 89,741 unique tracks for the prediction models. For the 720 tracks whose popularity differed across genre rows we kept the highest value. Instrumentalness, speechiness, and duration are strongly right skewed, so we added log transformed versions of each. The transform corrected duration (skewness from 11.1 to -0.3) but only partly helped speechiness (4.6 to 3.7) and barely changed instrumentalness (1.6 to 1.5), whose values are concentrated near zero; tree based models are unaffected by this skew. We defined a hit as popularity at least 60, which is the 90th percentile and yields a hit rate of 10.9 percent. Figures 1 to 3 show the popularity distribution, the feature skew, and the correlation structure; popularity is roughly symmetric apart from 10.4 percent of songs at exactly 0, and no single feature correlates with popularity above about 0.13 in absolute value.

**Prediction models.** On the unique song table we modeled popularity two ways. For the continuous score we fit linear regression, ridge, lasso, random forest, and gradient boosting, each against a mean predicting baseline. For the hit label we fit logistic regression, random forest, and gradient boosting against a majority class baseline, using balanced class weights to handle the 89 to 11 imbalance (sample weights for gradient boosting, which lacks a class weight parameter). To avoid leakage from duplicate tracks and from scaling, we split by track id with grouped five fold cross validation so no song appears in both training and testing, and standardized features fitting on the training fold only. We report RMSE and R squared for regression and accuracy, F1, and ROC AUC for classification, each on a held out test set.

**Genre analysis and tests.** On the song by genre table we fit one linear model of popularity on the audio features within each genre and compared the spread of per genre R squared values against a single pooled model, to test whether the audio to popularity relationship is genre dependent. We tested whether explicit tracks differ in popularity with a Welch t test, reporting Cohen's d with a 95 percent confidence interval, and repeated the test within each genre controlling the false discovery rate with the Benjamini-Hochberg procedure. Finally we standardized the audio features, ran k means clustering, and compared the clusters to the 114 genre labels with the adjusted Rand index.

## Results

**Audio features are weak predictors of popularity.** Every model beat its baseline but the ceiling was low (Figure 4). The best regressor, random forest, reached a test R squared of only 0.127, with gradient boosting close behind at 0.118 and the linear family near 0.035. The actual versus predicted plot (Figure 8) shows the models compress predictions toward the mean, as expected when the signal is weak. By feature importance (Figures 5 and 6), acousticness, instrumentalness, duration, danceability, and valence contributed most, while key, mode, and time signature contributed almost nothing.

**Hit classification is better than chance but modest.** The best classifier, gradient boosting with balanced weights, reached a test ROC AUC of 0.717, with random forest at 0.711 and logistic regression at 0.652 (Figure 7). On the hit class it recovered most hits (recall about 0.72) but at the cost of many false positives, giving an F1 near 0.28; the confusion matrix (Figure 9) shows this tradeoff, which is expected given the class imbalance.

**The audio to popularity link depends strongly on genre.** A single pooled linear model explains almost none of the variation in popularity (R squared 0.023), but fitting one model per genre reveals wide variation: per genre R squared ranges from 0.015 to 0.318 with a median near 0.068 (Figure 10). Audio features predict popularity relatively well inside genres such as sleep (0.318), comedy (0.307), and world music (0.296), and essentially not at all inside latin (0.018) and j-idol (0.015).

**Explicit songs are slightly more popular, unevenly across genres.** Explicit tracks have a higher mean popularity than non explicit tracks overall (36.9 versus 32.9), highly significant by a Welch t test (t = 15.2, p < 0.001) but small in magnitude (Cohen's d = 0.20, 95 percent CI 0.17 to 0.22). Within genres, of the 83 with enough songs in both groups, 35 were significant after Benjamini-Hochberg correction, 28 with explicit songs more popular and 7 less popular (Figure 11).

**Audio features do not recover genre.** K means clustering of the audio features barely matched the genre labels (adjusted Rand index 0.019 at ten clusters, 0.037 at 114). The projection in Figure 12 shows heavily overlapping clusters with no clean genre boundaries.

## Discussion

Our results support a clear and somewhat humbling conclusion: a song's measurable audio qualities explain very little about its popularity on their own. This holds across model families and matches our preliminary finding that no single feature correlated with popularity above about 0.13. It is consistent with popularity being driven largely by factors absent from the data, such as marketing, playlist placement, artist following, and release timing. Rather than treating this weak ceiling as a failure, we read it as the substantive finding: intrinsic sound is not what separates popular tracks from the rest.

The genre analysis refines that picture. The wide spread of per genre R squared values shows audio features matter much more in some genres than others, so the better question is not whether sound predicts popularity but where. Genres where listening context is tied to sonic qualities, such as sleep and ambient styles, are the most predictable from audio alone, while mainstream vocal genres are nearly unpredictable from sound. The explicit content test tells a parallel story: a small average effect masks real genre to genre variation that only emerges once we test within genres and correct for multiple comparisons. The clustering result confirms that genres overlap heavily in audio space, which both explains why genre specific modeling helps and cautions against treating audio features as a proxy for genre. The hit classifier is the most practically useful artifact: even with weak features it ranks tracks by hit probability well above chance, which could support coarse triage, though its false positive rate limits confident single track calls.

The main limitation is that our conclusions are bounded by the features available: we can show that intrinsic audio is a weak and uneven predictor, but we cannot measure the external drivers that likely dominate popularity. Future work could join this data with release date, artist level popularity, or playlist exposure to test those mechanisms directly.

## Statement of contributions

**Tejashree Venkatesh:** data acquisition and cleaning (code/01_clean.py), exploratory figures (Figures 1 to 3), repository and README, and the Introduction and data portion of Methods.

**Chaitanya Sandeep Jumale:** prediction and classification models (code/02_models.py), model figures (Figures 4 to 9), and the modeling portion of Methods and the model Results.

**Prasad Chandrashekhar Kanade:** genre analysis, hypothesis testing, and clustering (code/03_genre_tests.py), genre figures (Figures 10 to 12), and the Abstract, genre and test Results, Discussion, and References.

## References

[1] M. Pandya, Spotify Tracks Dataset, about 114,000 tracks across 114 genres, collected via the Spotify Web API. Available: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset

[2] Spotify for Developers, Web API Reference: Audio Features. Available: https://developer.spotify.com/documentation/web-api

[3] Y. Benjamini and Y. Hochberg, Controlling the false discovery rate: a practical and powerful approach to multiple testing, Journal of the Royal Statistical Society Series B, 57(1), 1995, pp. 289 to 300.

[4] L. Hubert and P. Arabie, Comparing partitions, Journal of Classification, 2(1), 1985, pp. 193 to 218.
