import whisper
import json
import os
from datetime import datetime

# -------------------- Data Management --------------------

def load_database(db_path: str = "metadata.json") -> dict:
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_database(database: dict, db_path: str = "database.json") -> None:
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    print(f"Database saved: {db_path}")


def add_session_to_database(database: dict, metadata: dict, audio_filename: str, session_data: dict) -> dict:
    category = metadata["category"]
    database.setdefault("metadata", {})
    database["metadata"].setdefault(category, {})
    database["metadata"][category][audio_filename] = session_data
    
    return database


# -------------------- Transcription --------------------

def transcribe_with_whisper(audio_path: str, machine_category) -> dict:
    """
    Returns text + segments (with timestamps) + language + duration.

    Whisper tries to naturally break segments at:
    ✔ pauses
    ✔ punctuation
    ✔ intonation boundaries
    ✔ breath sounds
    ✔ sentence endings
    """
    model = whisper.load_model("small")  # or "medium" 
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path, verbose=False, initial_prompt=f" This is about: {machine_category}")

    segments = []
    for seg in result["segments"]:
        segment_dict = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        }
        segments.append(segment_dict)
    
    combined_segments = combine_segments(segments, max_words=120)

    return {
        "text": result["text"],
        "segments": combined_segments,
        "language": result.get("language"),
        "duration": result.get("duration"),
    }


def combine_segments(segments: list, max_words: int = 120) -> list:
    """
    Combine segments into chunks that don't exceed max_words.
    Returns a list of dicts with 'text', 'start', and 'end' timestamps.
    """
    if not segments:
        return []
    
    combined_segments = []
    current_chunk = ""
    chunk_start = None
    chunk_end = None
    
    for seg in segments:
        text = seg["text"].strip()
        word_count_current = len(current_chunk.split()) if current_chunk else 0
        word_count_new = len(text.split())
        
        # If adding this segment would exceed the word limit
        if word_count_current + word_count_new > max_words:
            # Save the current chunk if it's not empty
            if current_chunk:
                combined_segments.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": current_chunk.strip()
                })
            # Start a new chunk with the current segment
            current_chunk = text
            chunk_start = seg["start"]
            chunk_end = seg["end"]
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + text
                chunk_end = seg["end"]  # Update end time to this segment's end
            else:
                # First segment in this chunk
                current_chunk = text
                chunk_start = seg["start"]
                chunk_end = seg["end"]
    
    # Don't forget the last chunk!
    if current_chunk:
        combined_segments.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": current_chunk.strip()
        })
    
    return combined_segments


# -------------------- Main --------------------

def transcribe():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Project root = parent folder of SpeachTT (where whispertest.py is)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    print("PROJECT_ROOT =", PROJECT_ROOT)
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    DB_PATH = os.path.join(DATA_ROOT, "database.json")
    database = load_database(DB_PATH)

    audio_file_name = "Video3.mp3"#input("Audio file: ").strip()
    audio_file = os.path.join(DATA_ROOT, "audio", audio_file_name)
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return

    metadata = {
        "name": "Markus",#input("Name: ").strip(),
        "category": "Brakes",#input("Category: ").strip(),
        "date": datetime.today().strftime('%Y-%m-%d')
    }
    
    audio_filename = os.path.basename(audio_file)
    transcriptionData = transcribe_with_whisper(audio_file, metadata["category"])

    session = {
        "name": metadata["name"],
        "audio_file": audio_file_name,
        "date": metadata["date"],
        "transcription": transcriptionData,
    }

    database = add_session_to_database(database, metadata, audio_filename, session)
    save_database(database, DB_PATH)

    print("\n" + "=" * 60)
    print("Transcription complete!")

    return [s["text"] for s in transcriptionData["segments"]]
