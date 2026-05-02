"""
FaceAuth Pro Backend — Cloud Version
- Serves the frontend HTML directly (no separate server needed)
- Reads config from environment variables
- Gunicorn-ready (no app.run needed in prod)
- Compatible with Render.com / Railway / Fly.io
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64, os, json, uuid, shutil
import numpy as np
import cv2
from deepface import DeepFace
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "faceauth_2026")
CORS(app)

# ── Config from environment variables ────────────────────────────────────────
# On Render: set these in Dashboard → Environment
DATA_DIR      = os.environ.get("DATA_DIR", "/app/data")   # use persistent disk path
DATASET_DIR   = os.path.join(DATA_DIR, "dataset")
LOG_FILE      = os.path.join(DATA_DIR, "access_log.json")
BLOCKED_FILE  = os.path.join(DATA_DIR, "blocked_users.json")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")
ADMIN_USER    = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS    = os.environ.get("ADMIN_PASS", "admin123")
MODEL_NAME    = "ArcFace"
DETECTOR      = "opencv"
THRESHOLD     = 0.40

os.makedirs(DATASET_DIR, exist_ok=True)

# ── Preload cascades ──────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Embedding cache ───────────────────────────────────────────────────────────
embedding_cache = {}

def build_embedding_cache():
    global embedding_cache
    embedding_cache = {}
    if not os.path.exists(DATASET_DIR):
        return
    for person in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(folder) or person.startswith("."):
            continue
        embs = []
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(folder, fname)
            try:
                result = DeepFace.represent(
                    img_path=path, model_name=MODEL_NAME,
                    detector_backend=DETECTOR, enforce_detection=False
                )
                if result:
                    embs.append(np.array(result[0]["embedding"]))
            except Exception as e:
                print(f"  Skip {path}: {e}")
        if embs:
            embedding_cache[person] = embs
            print(f"  Cached {len(embs)} embedding(s) for '{person}'")

def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return 1 - np.dot(a, b)

def preload_model():
    print("⏳ Preloading ArcFace model...")
    try:
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        DeepFace.represent(img_path=dummy, model_name=MODEL_NAME,
                           detector_backend=DETECTOR, enforce_detection=False)
        print("✓ ArcFace model loaded")
    except Exception as e:
        print(f"  Model preload warning: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_blocked():
    if os.path.exists(BLOCKED_FILE):
        with open(BLOCKED_FILE) as f:
            return set(json.load(f))
    return set()

def save_blocked(b):
    with open(BLOCKED_FILE, "w") as f:
        json.dump(list(b), f)

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {}

def save_profiles(p):
    with open(PROFILES_FILE, "w") as f:
        json.dump(p, f, indent=2)

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []

def append_log(entry):
    logs = load_logs()
    logs.insert(0, entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs[:500], f, indent=2)

def decode_image(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def preprocess(img):
    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# ── Serve Frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("face_auth_frontend.html")

# ── Routes: Register ──────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    data     = request.json
    username = data.get("username", "").strip()
    images   = data.get("images", [])

    if not username:
        return jsonify({"success": False, "message": "Username required."}), 400
    if not images:
        return jsonify({"success": False, "message": "No images provided."}), 400

    blocked = load_blocked()
    if username in blocked:
        return jsonify({"success": False, "message": "User is blocked."}), 403

    folder = os.path.join(DATASET_DIR, username)
    os.makedirs(folder, exist_ok=True)

    saved = 0
    new_embs = []
    for b64 in images:
        img = decode_image(b64)
        if img is None:
            continue
        img_proc = preprocess(img)
        count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
        path  = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(path, img_proc)
        try:
            result = DeepFace.represent(
                img_path=path, model_name=MODEL_NAME,
                detector_backend=DETECTOR, enforce_detection=False
            )
            if result:
                new_embs.append(np.array(result[0]["embedding"]))
                saved += 1
        except Exception:
            os.remove(path)

    if saved == 0:
        return jsonify({"success": False, "message": "No face detected. Face camera directly in good light."}), 400

    if username in embedding_cache:
        embedding_cache[username].extend(new_embs)
    else:
        embedding_cache[username] = new_embs

    append_log({"type": "REGISTER", "user": username, "time": datetime.now().isoformat(), "images": saved})
    return jsonify({"success": True, "message": f"User '{username}' registered with {saved} image(s)."})

# ── Routes: Verify ────────────────────────────────────────────────────────────

@app.route("/api/verify", methods=["POST"])
def verify():
    data       = request.json
    image_b64  = data.get("image")
    frames_b64 = data.get("frames", [])
    session_id = data.get("session_id", "default")

    if not image_b64:
        return jsonify({"success": False, "message": "Image required."}), 400
    if not embedding_cache:
        return jsonify({"success": False, "message": "No registered users."}), 404

    movement       = False
    face_motions   = []
    bg_motions     = []
    texture_scores = []
    prev_gray      = None

    for b64 in frames_b64:
        img = decode_image(b64)
        if img is None:
            continue
        gray = cv2.cvtColor(cv2.resize(img, (160, 120)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gh, gw = gray.shape
        face_region = gray[gh//4: gh*3//4, gw//4: gw*3//4]
        bg_top      = gray[0:gh//6, :]
        bg_bottom   = gray[gh*5//6:, :]
        bg_left     = gray[:, 0:gw//6]
        bg_right    = gray[:, gw*5//6:]
        lap_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        texture_scores.append(lap_var)
        if prev_gray is not None:
            prev_face = prev_gray[gh//4: gh*3//4, gw//4: gw*3//4]
            prev_bg_t = prev_gray[0:gh//6, :]
            prev_bg_b = prev_gray[gh*5//6:, :]
            prev_bg_l = prev_gray[:, 0:gw//6]
            prev_bg_r = prev_gray[:, gw*5//6:]
            face_diff = cv2.absdiff(face_region, prev_face).mean()
            bg_diff   = (cv2.absdiff(bg_top, prev_bg_t).mean() +
                         cv2.absdiff(bg_bottom, prev_bg_b).mean() +
                         cv2.absdiff(bg_left, prev_bg_l).mean() +
                         cv2.absdiff(bg_right, prev_bg_r).mean()) / 4.0
            face_motions.append(face_diff)
            bg_motions.append(bg_diff)
        prev_gray = gray

    if len(face_motions) >= 2:
        avg_face = sum(face_motions) / len(face_motions)
        avg_bg   = sum(bg_motions)   / len(bg_motions)
        avg_tex  = sum(texture_scores) / len(texture_scores)
        max_face = max(face_motions)
        ratio    = avg_face / (avg_bg + 0.1)
        if avg_tex < 30:
            movement = False
        elif ratio < 1.2 and avg_bg > 1.5:
            movement = False
        elif avg_face > 1.8 and ratio > 1.3:
            movement = True
        elif max_face > 7.0 and avg_bg < 3.0:
            movement = True

    if not movement:
        return jsonify({
            "success": True, "authenticated": False, "liveness_passed": False,
            "message": "Liveness failed — please move or blink slightly."
        })

    img = decode_image(image_b64)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image."}), 400
    img_proc = preprocess(img)

    try:
        result = DeepFace.represent(
            img_path=img_proc, model_name=MODEL_NAME,
            detector_backend=DETECTOR, enforce_detection=False
        )
        if not result:
            return jsonify({"success": True, "authenticated": False, "liveness_passed": True,
                            "message": "No face detected in frame."})
        query_emb = np.array(result[0]["embedding"])
    except Exception as e:
        return jsonify({"success": False, "message": f"Embedding failed: {str(e)}"}), 500

    blocked = load_blocked()
    results = []
    for username, embs in embedding_cache.items():
        distances = [cosine_distance(query_emb, e) for e in embs]
        best_dist = min(distances)
        conf      = round(max(0, (1 - best_dist) * 100), 1)
        results.append({
            "username":   username,
            "distance":   round(float(best_dist), 4),
            "confidence": float(conf),
            "verified":   bool(best_dist < THRESHOLD)
        })

    results.sort(key=lambda x: x["distance"])
    best = results[0] if results else None

    if best and best["verified"]:
        name = best["username"]
        conf = best["confidence"]
        if name in blocked:
            append_log({"type": "BLOCKED", "user": name, "time": datetime.now().isoformat(),
                        "confidence": conf, "ip": request.remote_addr})
            return jsonify({
                "success": True, "authenticated": False, "liveness_passed": True,
                "blocked": True, "message": f"User '{name}' is blocked.",
                "matched_user": name, "confidence": conf, "all_results": results
            })
        append_log({"type": "ACCESS_GRANTED", "user": name, "time": datetime.now().isoformat(),
                    "confidence": conf, "ip": request.remote_addr})
        profiles = load_profiles()
        return jsonify({
            "success": True, "authenticated": True, "liveness_passed": True,
            "matched_user": name, "confidence": conf,
            "threshold_used": round((1 - THRESHOLD) * 100),
            "profile": profiles.get(name, {}),
            "all_results": results
        })
    else:
        top = results[0] if results else {}
        append_log({"type": "ACCESS_DENIED", "user": "Unknown", "time": datetime.now().isoformat(),
                    "ip": request.remote_addr})
        return jsonify({
            "success": True, "authenticated": False, "liveness_passed": True,
            "message": "No matching identity found.",
            "closest_match": top.get("username"),
            "closest_confidence": top.get("confidence", 0),
            "all_results": results
        })

# ── Routes: Users ─────────────────────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
def list_users():
    blocked = load_blocked()
    users = []
    for name in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, name)
        if os.path.isdir(folder) and not name.startswith("."):
            count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
            users.append({"username": name, "images": count, "blocked": name in blocked})
    return jsonify({"success": True, "users": users, "count": len(users)})

@app.route("/api/users/<username>", methods=["DELETE"])
def delete_user(username):
    path = os.path.join(DATASET_DIR, username)
    if os.path.exists(path):
        shutil.rmtree(path)
    embedding_cache.pop(username, None)
    append_log({"type": "DELETE", "user": username, "time": datetime.now().isoformat()})
    return jsonify({"success": True, "message": f"'{username}' deleted."})

@app.route("/api/users/<username>/block", methods=["POST"])
def block_user(username):
    b = load_blocked(); b.add(username); save_blocked(b)
    append_log({"type": "BLOCK", "user": username, "time": datetime.now().isoformat()})
    return jsonify({"success": True, "message": f"'{username}' blocked."})

@app.route("/api/users/<username>/unblock", methods=["POST"])
def unblock_user(username):
    b = load_blocked(); b.discard(username); save_blocked(b)
    return jsonify({"success": True, "message": f"'{username}' unblocked."})

@app.route("/api/users/<username>/profile", methods=["GET"])
def get_profile(username):
    profiles = load_profiles()
    return jsonify({"success": True, "username": username, "profile": profiles.get(username, {})})

@app.route("/api/users/<username>/profile", methods=["POST"])
def save_profile(username):
    profiles = load_profiles()
    data = request.json
    profiles[username] = {
        "roll_no":    data.get("roll_no", ""),
        "department": data.get("department", ""),
        "year":       data.get("year", ""),
        "email":      data.get("email", ""),
        "phone":      data.get("phone", ""),
        "extra":      data.get("extra", "")
    }
    save_profiles(profiles)
    return jsonify({"success": True, "message": f"Profile saved for '{username}'."})

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.json
    if data.get("username") == ADMIN_USER and data.get("password") == ADMIN_PASS:
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Invalid credentials."}), 401

@app.route("/api/logs", methods=["GET"])
def get_logs():
    limit = int(request.args.get("limit", 100))
    logs  = load_logs()[:limit]
    return jsonify({"success": True, "logs": logs, "total": len(load_logs())})

@app.route("/api/logs/clear", methods=["DELETE"])
def clear_logs():
    with open(LOG_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"success": True})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": MODEL_NAME,
        "registered_users": len(embedding_cache),
        "total_embeddings": sum(len(v) for v in embedding_cache.values()),
        "threshold": THRESHOLD
    })

# ── Startup (for gunicorn) ────────────────────────────────────────────────────
preload_model()
print("⏳ Building embedding cache...")
build_embedding_cache()
print(f"✓ {len(embedding_cache)} user(s) cached | Ready.")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
