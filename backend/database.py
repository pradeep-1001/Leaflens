import os
import sqlite3
import json
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "leaflens.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        device_id TEXT NOT NULL,
        disease TEXT NOT NULL,
        confidence REAL NOT NULL,
        health_score INTEGER NOT NULL,
        necrosis_pct REAL NOT NULL,
        temperature_c REAL NOT NULL,
        humidity_pct REAL NOT NULL,
        lux REAL NOT NULL,
        leaf_wetness INTEGER NOT NULL,
        vpd_kpa REAL NOT NULL,
        image_path TEXT,
        gradcam_path TEXT,
        recommendations TEXT
    );
    """)
    conn.commit()
    conn.close()
    print(f"[OK] SQLite database initialized at: {DB_PATH}")

def save_scan(scan_id, device_id, disease, confidence, health_score, necrosis_pct,
              temperature_c, humidity_pct, lux, leaf_wetness, vpd_kpa,
              image_path, gradcam_path, recommendations):
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO scans (
        scan_id, timestamp, device_id, disease, confidence, health_score,
        necrosis_pct, temperature_c, humidity_pct, lux, leaf_wetness,
        vpd_kpa, image_path, gradcam_path, recommendations
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_id, now_iso, device_id, disease, confidence, health_score,
        necrosis_pct, temperature_c, humidity_pct, lux, 1 if leaf_wetness else 0,
        vpd_kpa, image_path, gradcam_path, json.dumps(recommendations)
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_recent_scans(limit=30):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        item = dict(r)
        item["leaf_wetness"] = bool(item["leaf_wetness"])
        try:
            item["recommendations"] = json.loads(item["recommendations"])
        except Exception:
            item["recommendations"] = []
        results.append(item)
    return results

def get_analytics_summary():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total_scans, AVG(health_score) as avg_health FROM scans")
    totals = cursor.fetchone()
    
    cursor.execute("SELECT disease, COUNT(*) as count FROM scans GROUP BY disease")
    disease_counts = {r["disease"]: r["count"] for r in cursor.fetchall()}
    conn.close()

    return {
        "total_scans": totals["total_scans"] if totals else 0,
        "average_health_score": round(totals["avg_health"], 1) if totals and totals["avg_health"] else 100.0,
        "disease_distribution": disease_counts
    }

if __name__ == "__main__":
    init_db()
