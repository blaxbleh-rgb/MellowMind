import os
import random
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from sklearn.neighbors import NearestNeighbors
from werkzeug.security import check_password_hash, generate_password_hash

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
DB_PATH = BACKEND_DIR / "songs.db"
MODEL_PATH = BACKEND_DIR / "model.joblib"

MIN_PASSWORD_LENGTH = 8
MIN_LIKES = 3       # likes needed before recommendations start matching taste
TOP_N = 5000        # best-fitting songs for a mood to choose from
POOL_SIZE = 1500    # random subset of those that gets ranked by taste
HISTORY_SIZE = 10   # songs shown on the History page
HISTORY_KEEP = 200  # newest history events stored per user


# ---------------------------------------------------------------- setup

def load_secret_key():
    """Key used to sign login cookies. Kept in secret.key so logins survive restarts."""
    from_env = os.environ.get("FLASK_SECRET_KEY")
    if from_env:
        return from_env
    key_file = BACKEND_DIR / "secret.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    return key_file.read_text().strip()


app = Flask(__name__, static_folder=None)
app.secret_key = load_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

# The trained model from train_model.py (None until that script has been run)
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the tables for accounts and feedback (songs come from load_data.py, scores from train_model.py)."""
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS feedback (
                user_id INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('like', 'skip')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, track_id)
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                event TEXT NOT NULL CHECK (event IN ('like', 'skip', 'search')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id, id);
        """)


init_db()


# ---------------------------------------------------------------- taste ("more like this")

def scale(rows):
    """Put songs' audio values on a common scale, using the scaler saved by train_model.py."""
    values = np.array([[row[name] for name in MODEL["features"]] for row in rows], dtype=float)
    return MODEL["scaler"].transform(values)


def user_taste(conn, user_id):
    """Scaled audio values of the songs this user liked and skipped.
    Liked is None until they have MIN_LIKES likes; skipped is None if they skipped nothing."""
    features = ", ".join(MODEL["features"])

    def fetch(action):
        rows = conn.execute(
            f"""SELECT {features} FROM songs
                WHERE track_id IN (SELECT track_id FROM feedback WHERE user_id = ? AND action = ?)""",
            (user_id, action),
        ).fetchall()
        return rows

    liked, skipped = fetch("like"), fetch("skip")
    return (
        scale(liked) if len(liked) >= MIN_LIKES else None,
        scale(skipped) if skipped else None,
    )


def rank_by_taste(candidates, liked, skipped):
    """Order candidate songs by how close each is to the nearest song the user liked.

    Using the nearest liked song (not an average) keeps different tastes apart:
    someone who likes calm piano and loud dance music gets both, not a blurry middle.
    Songs that sit closer to a skipped song than to any liked song go to the back.
    """
    X = scale(candidates)
    distance_to_liked = NearestNeighbors(n_neighbors=1).fit(liked).kneighbors(X)[0][:, 0]
    order = np.argsort(distance_to_liked)
    if skipped is not None:
        distance_to_skipped = NearestNeighbors(n_neighbors=1).fit(skipped).kneighbors(X)[0][:, 0]
        closer_to_liked = distance_to_liked <= distance_to_skipped
        order = np.concatenate([order[closer_to_liked[order]], order[~closer_to_liked[order]]])
    return [candidates[i] for i in order]


# ---------------------------------------------------------------- accounts

def current_user():
    """The logged-in user's row, or None."""
    user_id = session.get("user_id")
    if user_id is None:
        return None
    with db() as conn:
        user = conn.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None:
        session.clear()
    return user


def start_session(user_id):
    session.clear()
    session["user_id"] = user_id
    session.permanent = True


def read_credentials():
    data = request.get_json(silent=True) or {}
    return str(data.get("email", "")).strip().lower(), str(data.get("password", ""))


@app.post("/api/register")
def register():
    email, password = read_credentials()
    if len(email) > 254 or "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Enter a valid email address."}), 400
    if not MIN_PASSWORD_LENGTH <= len(password) <= 128:
        return jsonify({"error": f"Password must be {MIN_PASSWORD_LENGTH} to 128 characters."}), 400
    try:
        with db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, generate_password_hash(password)),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists. Try logging in."}), 409
    start_session(user_id)
    return jsonify({"email": email}), 201


@app.post("/api/login")
def login():
    email, password = read_credentials()
    with db() as conn:
        user = conn.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Wrong email or password."}), 401
    start_session(user["id"])
    return jsonify({"email": email})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/me")
def me():
    user = current_user()
    if user is None:
        return jsonify({"error": "Not logged in."}), 401
    return jsonify({"email": user["email"]})


# ---------------------------------------------------------------- songs and feedback

def format_duration(ms):
    seconds = int(round(ms / 1000))
    return f"{seconds // 60}:{seconds % 60:02d}"


@app.get("/api/songs")
def songs():
    if MODEL is None:
        return jsonify({"error": "The mood model is missing. Run train_model.py first."}), 500

    mood = request.args.get("mood", "").lower()
    if mood not in MODEL["mood_blends"]:
        return jsonify({"error": f"Unknown mood. Use one of: {', '.join(MODEL['mood_blends'])}"}), 400
    limit = max(1, min(request.args.get("limit", 10, type=int), 50))
    user = current_user()

    columns = ", ".join(f"s.{name}" for name in MODEL["features"])
    # Logged-in users never see a song they already liked or skipped
    exclude = "AND m.track_id NOT IN (SELECT track_id FROM feedback WHERE user_id = ?)" if user else ""
    params = [user["id"]] if user else []

    try:
        with db() as conn:
            # The songs the model scored highest for this mood (mood is checked against the known moods above)
            candidates = conn.execute(
                f"""SELECT s.track_id, s.duration_ms, {columns}
                    FROM song_scores m JOIN songs s ON s.track_id = m.track_id
                    WHERE 1 = 1 {exclude}
                    ORDER BY m.mood_{mood} DESC LIMIT ?""",
                [*params, TOP_N],
            ).fetchall()
            liked, skipped = user_taste(conn, user["id"]) if user else (None, None)
    except sqlite3.OperationalError:
        return jsonify({"error": "Song scores are missing. Run train_model.py first."}), 500

    personalized = liked is not None
    if personalized:
        pool = random.sample(candidates, min(POOL_SIZE, len(candidates)))
        chosen = rank_by_taste(pool, liked, skipped)[:limit]
    else:
        chosen = random.sample(candidates, min(limit, len(candidates)))

    # The dataset has no titles or artists, only the Spotify track id.
    # The frontend shows each track with Spotify's embedded player.
    return jsonify([
        {"id": row["track_id"], "duration": format_duration(row["duration_ms"]), "personalized": personalized}
        for row in chosen
    ])


def record_history(conn, user_id, track_id, event):
    """Log that the user liked or skipped a song."""
    conn.execute(
        "INSERT INTO history (user_id, track_id, event) VALUES (?, ?, ?)",
        (user_id, track_id, event),
    )
    # Keep the table small: only the newest HISTORY_KEEP events per user
    conn.execute(
        """DELETE FROM history WHERE user_id = ?
           AND id NOT IN (SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?)""",
        (user_id, user_id, HISTORY_KEEP),
    )


@app.post("/api/feedback")
def feedback():
    user = current_user()
    if user is None:
        return jsonify({"error": "Log in first."}), 401

    data = request.get_json(silent=True) or {}
    track_id = str(data.get("track_id", ""))
    action = data.get("action")
    if not track_id or action not in ("like", "skip", "clear"):
        return jsonify({"error": "Send a track_id and an action: like, skip or clear."}), 400

    with db() as conn:
        if conn.execute("SELECT 1 FROM songs WHERE track_id = ?", (track_id,)).fetchone() is None:
            return jsonify({"error": "Unknown song."}), 404
        if action == "clear":
            conn.execute("DELETE FROM feedback WHERE user_id = ? AND track_id = ?", (user["id"], track_id))
        else:
            conn.execute(
                """INSERT INTO feedback (user_id, track_id, action) VALUES (?, ?, ?)
                   ON CONFLICT(user_id, track_id)
                   DO UPDATE SET action = excluded.action, created_at = CURRENT_TIMESTAMP""",
                (user["id"], track_id, action),
            )
            record_history(conn, user["id"], track_id, action)
    return jsonify({"ok": True})


@app.get("/api/history")
def history():
    """The last HISTORY_SIZE different songs the user liked or skipped, newest first."""
    user = current_user()
    if user is None:
        return jsonify({"error": "Log in first."}), 401
    with db() as conn:
        rows = conn.execute(
            """SELECT track_id, event, created_at FROM history
               WHERE id IN (SELECT MAX(id) FROM history
                            WHERE user_id = ? AND event IN ('like', 'skip') GROUP BY track_id)
               ORDER BY id DESC LIMIT ?""",
            (user["id"], HISTORY_SIZE),
        ).fetchall()
    return jsonify([
        {"id": row["track_id"], "event": row["event"], "at": row["created_at"].replace(" ", "T") + "Z"}
        for row in rows
    ])


# ---------------------------------------------------------------- frontend

@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
