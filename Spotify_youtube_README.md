# Spotify & YouTube Dataset Analysis

A full end-to-end data science project in Python that explores what makes a song successful across Spotify and YouTube — combining **exploratory data analysis**, **unsupervised clustering**, **stream count regression**, and **hit song classification** into one cohesive notebook.

---

## Business Problem

What separates a hit from a miss? This project answers that question by analyzing audio features (danceability, energy, valence, tempo, etc.) alongside real platform performance data (Spotify streams, YouTube views and likes). The goals are:

1. **Understand** how audio characteristics distribute across the music landscape  
2. **Segment** songs into meaningful clusters by their sonic fingerprint  
3. **Predict** how many streams a song will earn based on its audio features  
4. **Classify** whether a song will be a hit (top 25% of streams)

---

## Data Source

- **Dataset:** `Spotify_Youtube.csv`  
- Combined dataset of songs with Spotify audio analysis features and YouTube engagement metrics per track

| Feature | Description |
| :---- | :---- |
| `Artist` | Artist name |
| `Album_type` | Album / Single / Compilation |
| `Danceability` | How suitable a track is for dancing (0–1) |
| `Energy` | Perceptual intensity and activity (0–1) |
| `Loudness` | Overall loudness in decibels |
| `Speechiness` | Presence of spoken words (0–1) |
| `Acousticness` | Confidence the track is acoustic (0–1) |
| `Instrumentalness` | Likelihood of no vocals (0–1) |
| `Liveness` | Presence of a live audience (0–1) |
| `Valence` | Musical positiveness / happiness (0–1) |
| `Tempo` | Estimated BPM |
| `Duration_ms` | Track length in milliseconds |
| `Key` | Musical key (0–11) |
| `Stream` | Total Spotify streams |
| `Views` | Total YouTube views |
| `Likes` | Total YouTube likes |
| `Licensed` | Whether the YouTube video is licensed |
| `official_video` | Whether an official YouTube video exists |

## Methodology

### 1\. Data Preprocessing

- Dropped rows with missing values across all 13 key audio and engagement columns  
- Converted boolean-like string columns (`Licensed`, `official_video`) to proper Python booleans

### 2\. Exploratory Data Analysis (EDA)

- Histograms of all 6 core audio features  
- Log-scaled distributions of Streams and YouTube Views (heavy right skew)  
- Boxplots of Valence, Danceability, and Energy by Album type  
- Bar chart of song count by Album type  
- Top 10 artists by total Spotify streams and YouTube views  
- Correlation heatmap across all numerical features  
- Scatter plots: Valence vs Streams, Danceability vs Streams

### 3\. K-Means Clustering (Unsupervised)

- Standardized 11 audio features with `StandardScaler`  
- Elbow Method to determine optimal k → **k \= 4 clusters**  
- Analyzed each cluster by mean valence, mean stream count, and dominant album type  
- PCA (2 components) for 2D cluster visualization

### 4\. Stream Count Prediction (Regression)

Predicting `log(Streams + 1)` from 11 audio features using:

- **Linear Regression** — interpretable baseline  
- **Random Forest Regressor** — ensemble, captures non-linearity  
- **Gradient Boosting Regressor** — sequential boosting, best fit

Models evaluated on R² and RMSE. Feature importance extracted from Random Forest.

### 5\. Hit Song Classification

- Defined a **"hit"** as any song in the top 25% of streams (threshold: 75th percentile)  
- Binary classification using:  
  - **Decision Tree** (max depth \= 4\) — fully visualizable, interpretable rules  
  - **Random Forest Classifier** — higher accuracy ensemble  
- Evaluated via classification report, confusion matrix, and ROC / AUC curves  
- Feature importance ranking for hit prediction

---

## Results

### Regression — Predicting Stream Count

| Model | R² | RMSE |
| :---- | :---- | :---- |
| Linear Regression | 0.0725 | 1.5947 |
| **Random Forest Regressor** | **0.2193** | **1.4630** |
| Gradient Boosting | 0.1354 | 1.5396 |

Random Forest achieves the best R² (0.2193) and lowest RMSE (1.4630) for predicting log-transformed stream counts. The modest R² across all models reflects a real-world truth: audio features alone are weak predictors of streams — artist fame, playlist placement, and marketing explain most of the variance the model can't capture.

### Classification — Hit Song Prediction

| Model | Accuracy | Hit Precision | Hit Recall | Hit F1 |
| :---- | :---- | :---- | :---- | :---- |
| Decision Tree | 0.75 | 0.00 | 0.00 | 0.00 |
| **Random Forest** | **0.79** | **0.86** | **0.19** | **0.31** |

The Decision Tree (max depth \= 4\) fails entirely on the minority "Hit" class — it predicts Non-Hit for everything, hitting 75% accuracy only because 75% of songs are non-hits by definition. The Random Forest is far more useful: **86% precision** on hit prediction means when it calls a song a hit, it's right 86% of the time. The tradeoff is conservative recall (0.19) — it misses many hits, but its positive calls are highly reliable.

---

## 💡 Business Solution: Will This Song Be a Hit?

Based on feature importance from the Random Forest classifier and correlation analysis:

### 🎯 The Hit Song Formula

| Audio Feature | Hit Profile | Why It Matters |
| :---- | :---- | :---- |
| **Danceability** | High (0.65–0.85) | Strongest positive correlation with stream count |
| **Energy** | High (0.60–0.85) | High-energy tracks feel engaging and radio-friendly |
| **Loudness** | –8 dB to –4 dB | Louder, polished production signals mainstream appeal |
| **Valence** | Moderate (0.40–0.75) | Neither too dark nor too upbeat — broadly relatable |
| **Acousticness** | Low (below 0.25) | Produced/electronic sound dominates streaming charts |
| **Speechiness** | Low (below 0.15) | Unless rap/hip-hop, high speechiness limits audience reach |
| **Tempo** | 100–140 BPM | Sweet spot for pop and dance — maximizes accessibility |

### 🎵 Cluster Insights

K-Means found 4 distinct song archetypes. The **highest-streaming cluster** is defined by:

- High danceability \+ high energy \+ moderate valence  
- Predominantly released as **singles**  
- Low acousticness — produced, not stripped-down

The **lowest-streaming cluster** skews acoustic, low-energy, and high-valence — quality songs with a smaller commercial ceiling.

### ✅ Actionable Takeaways for Artists & Labels

- **Release as a single** — singles consistently outperform album tracks on both platforms  
- **Prioritize danceability and energy** — these are the top predictors in the hit classifier  
- **Pair with an official YouTube video** — `official_video = True` tracks show substantially higher views and likes  
- A song scoring danceability \> 0.70, energy \> 0.65, and acousticness \< 0.20 sits in the highest-performing cluster and has the strongest hit potential in this dataset

Note: These are data-driven signals, not guarantees. Artist recognition, playlist placement, and release timing also drive success in ways not captured by audio features alone.  
