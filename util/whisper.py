import whisper
import json
import os
import util.database as db
from util import paths


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

def transcribe(audio_file_name, database, metadata):

    audio_file = os.path.join(paths.DATA_ROOT, "audio", audio_file_name)
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return


    audio_filename = os.path.basename(audio_file)
    transcriptionData = transcribe_with_whisper(audio_file, metadata["category"])

    session = {
        "name": metadata["name"],
        "audio_file": audio_file_name,
        "date": metadata["date"],
        "transcription": transcriptionData,
    }

    database = db.add_session(database, metadata, session)
    db.save(database, paths.DATABASE)

    print("\n" + "=" * 60)
    print("Transcription complete!")

    return [s["text"] for s in transcriptionData["segments"]], metadata["name"], metadata["date"]
