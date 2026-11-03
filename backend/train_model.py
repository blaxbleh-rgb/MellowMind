"""Trains Mellowmind's mood model with scikit-learn and scores every song.

Run from the backend folder, after load_data.py:
    python train_model.py

What it does:
  1. Trains a classifier that predicts a song's emotion (the dataset's labels 0-3) from its audio values.
  2. Prints how accurate it is on songs it has never seen (also saved to model_report.txt).
  3. Uses the model's probabilities to score every song for each app mood (sunny, tender, ...).
  4. Saves the model to model.joblib and the scores to the song_scores table in songs.db.
"""
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "songs.db"
MODEL_PATH = BACKEND_DIR / "model.joblib"
REPORT_PATH = BACKEND_DIR / "model_report.txt"

# The audio values the model learns from (and that "more like this" compares)
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]

# What the dataset's labels mean, worked out from their average audio values:
# 0 = lowish energy and valence, fairly acoustic; 1 = energetic and positive; 2 = very high energy;
# 3 = very quiet and acoustic. Only used for readable names, the model just learns the numbers 0-3.
LABEL_NAMES = {0: "sad", 1: "happy", 2: "energetic", 3: "calm"}

# Each app mood is a weighted blend of the model's emotion probabilities (p_...)
# and a few raw audio values (all between 0 and 1). Tune these freely and re-run this script.
MOOD_BLENDS = {
    "sunny": {"p_happy": 1.0},
    "tender": {"p_calm": 0.4, "p_sad": 0.4, "acousticness": 0.2},
    "focused": {"p_calm": 0.6, "instrumentalness": 0.4},
    "restless": {"p_energetic": 1.0},
    "lowkey": {"p_calm": 0.6, "p_sad": 0.4},
    # The dataset has no release year, so "nostalgic" is only an approximation: warm and acoustic.
    "nostalgic": {"p_sad": 0.3, "p_happy": 0.3, "acousticness": 0.4},
}

TOP_N = 5000  # only used for the summary printed at the end


def main():
    if not DB_PATH.exists():
        sys.exit("songs.db not found. Run load_data.py first.")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT track_id, label, {', '.join(FEATURES)} FROM songs", conn)
    print(f"Loaded {len(df):,} songs")

    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training the model (about a minute)...")
    model = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # ---- How good is it? Judged on songs the model never saw while training
    predictions = model.predict(X_test)
    names = [LABEL_NAMES[c] for c in model.classes_]
    accuracy = accuracy_score(y_test, predictions)
    baseline = y_test.value_counts(normalize=True).max()
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=[f"true {n}" for n in names], columns=[f"pred {n}" for n in names],
    )
    sample = X_test.sample(min(10000, len(X_test)), random_state=0)
    importance = permutation_importance(model, sample, y_test.loc[sample.index], n_repeats=3, random_state=0)
    importance = pd.Series(importance.importances_mean, index=FEATURES).sort_values(ascending=False).round(3)

    report = "\n".join([
        f"Accuracy on {len(X_test):,} unseen songs: {accuracy:.1%}",
        f"(Always guessing the most common emotion would score {baseline:.1%})",
        "",
        classification_report(y_test, predictions, target_names=names, digits=3),
        "Confusion matrix (rows = real emotion, columns = predicted):",
        matrix.to_string(),
        "",
        "Which audio values matter most (drop in accuracy when shuffled):",
        importance.to_string(),
    ])
    print("\n" + report + "\n")
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    # ---- Score every song: emotion probabilities, then one score per app mood
    proba = model.predict_proba(X)
    scores = pd.DataFrame({"track_id": df["track_id"]})
    for i, cls in enumerate(model.classes_):
        scores[f"p_{LABEL_NAMES[cls]}"] = proba[:, i]
    available = pd.concat([scores, df[FEATURES]], axis=1)

    for mood, blend in MOOD_BLENDS.items():
        unknown = [name for name in blend if name not in available.columns]
        if unknown:
            sys.exit(f"Mood '{mood}' uses unknown columns: {unknown}")
        scores[f"mood_{mood}"] = sum(weight * available[name] for name, weight in blend.items())
    scores = scores.round(4)

    with sqlite3.connect(DB_PATH) as conn:
        scores.to_sql("song_scores", conn, if_exists="replace", index=False, chunksize=20000)
        conn.execute("CREATE UNIQUE INDEX idx_scores_track ON song_scores(track_id)")
        for mood in MOOD_BLENDS:
            conn.execute(f"CREATE INDEX idx_scores_{mood} ON song_scores(mood_{mood})")

    # The scaler puts every audio value on the same scale, so "more like this" compares them fairly
    scaler = StandardScaler().fit(X.to_numpy())
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
            "label_names": LABEL_NAMES,
            "mood_blends": MOOD_BLENDS,
            "test_accuracy": float(accuracy),
        },
        MODEL_PATH,
    )

    # ---- Sanity check: what do the best songs for each mood sound like?
    print(f"Average sound of each mood's top {TOP_N:,} songs (check these make sense):")
    rows = {}
    for mood in MOOD_BLENDS:
        top = scores.nlargest(TOP_N, f"mood_{mood}")["track_id"]
        rows[mood] = df[df["track_id"].isin(top)][["energy", "valence", "acousticness", "instrumentalness", "tempo"]].mean()
    print(pd.DataFrame(rows).T.round(2).to_string())
    print(f"\nSaved {MODEL_PATH.name}, scores in songs.db, and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()