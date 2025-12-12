import json
import os


def load_database(db_path: str = "metadata.json") -> dict:
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(database: dict, db_path: str = "database.json") -> None:
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    print(f"Database saved: {db_path}")


def add_session(database: dict, metadata: dict, session_data: dict) -> dict:
    category = metadata["category"]
    database.setdefault("metadata", {})
    database["metadata"].setdefault(category, {})
    database["metadata"][category][session_data["audio_file"]] = session_data
    return database