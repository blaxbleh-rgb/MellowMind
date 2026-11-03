import sqlite3
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
CSV_PATH = BACKEND_DIR.parent / "data" / "278k_labelled_uri.csv"
DB_PATH = BACKEND_DIR / "songs.db"

# Audio columns we keep from the CSV
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


def main():
    if not CSV_PATH.exists():
        sys.exit(f"Could not find {CSV_PATH}\nPut 278k_labelled_uri.csv inside the data folder.")

    # If the rows have one more field than the header (a leftover row number),
    # pandas automatically uses that first field as the index, so the columns line up.
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"duration (ms)": "duration_ms", "labels": "label"})

    needed = ["duration_ms", *FEATURES, "label", "uri"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"Columns missing from the CSV: {missing}\nFound: {list(df.columns)}")

    df = df[needed].dropna()

    # Sanity check: energy and valence should be between 0 and 1.
    # If they are not, the columns are probably shifted by one.
    for col in ("energy", "valence", "danceability"):
        share_ok = df[col].between(0, 1).mean()
        if share_ok < 0.99:
            sys.exit(
                f"Column '{col}' has values outside 0-1, so the columns look shifted.\n"
                f"First rows:\n{df.head()}\nSend me this output and I'll fix the loader."
            )

    # "spotify:track:3v6sBj3swihU8pXQQHhDZo" -> "3v6sBj3swihU8pXQQHhDZo"
    df["track_id"] = df["uri"].str.split(":").str[-1]
    df = df.drop(columns=["uri"]).drop_duplicates(subset="track_id")
    df["label"] = df["label"].astype(int)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("songs", conn, if_exists="replace", index=False, chunksize=20000)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_track ON songs(track_id)")

    print(f"Saved {len(df):,} songs to {DB_PATH}")

    if "--check" in sys.argv:
        print("\nAverage audio values per label (use this to work out what each label means):")
        print(df.groupby("label")[["energy", "valence", "acousticness", "tempo"]].mean().round(2))
        print("\nSongs per label:")
        print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
