from __future__ import annotations

import os
import time
import base64
import tempfile
from datetime import datetime
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from openai import OpenAI
from playsound import playsound
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------
WATCH_FOLDER = Path.home() / "Pictures"         # folder to watch for images
AUDIO_FOLDER = WATCH_FOLDER / "spoken"          # folder to keep MP3s  ← NEW
PROMPT_TEXT = (
    "You are a friendly assistant. "
    "Look at the image and give the user concise, constructive feedback."
)
MODEL_VISION = "gpt-4o-mini"
MODEL_TTS = "tts-1"
VOICE = "alloy"

# Make sure the audio folder exists  ← NEW
AUDIO_FOLDER.mkdir(exist_ok=True)

# Instantiate a single OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()


def talk(text: str) -> None:
    """Convert text to speech, save it, play it, and tell the user where it lives."""
    resp = client.audio.speech.create(
        model=MODEL_TTS,
        voice=VOICE,
        input=text,
        response_format="mp3",
    )

    # Timestamped filename so nothing is ever overwritten  ← CHANGED
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    mp3_path = AUDIO_FOLDER / f"reply-{ts}.mp3"

    with open(mp3_path, "wb") as f:
        f.write(resp.content)

    playsound(mp3_path)            # play the file we just saved
    print(f"🔊  Saved speech to {mp3_path}")  # show where it was stored  ← NEW


def ask_vision(image_path: str) -> str:
    """Send the image plus prompt to the Vision model and return the reply text."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    extension = Path(image_path).suffix.lstrip(".")
    data_url = f"data:image/{extension};base64,{img_b64}"

    response = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()


class ImageDropHandler(FileSystemEventHandler):
    """Watch for newly created image files and process them."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

    def on_created(self, event) -> None:  # type: ignore
        if event.is_directory:
            return

        ext = Path(event.src_path).suffix.lower()
        if ext not in self.IMAGE_EXTS:
            return

        # Slight delay to ensure the file is fully written
        time.sleep(0.5)
        try:
            print(f"🔍  New image detected: {event.src_path}")
            answer = ask_vision(event.src_path)
            print(f"🗨️  Model says: {answer}")
            talk(answer)
        except Exception as err:  # noqa: BLE001
            print(f"❌  Error processing {event.src_path}: {err}")


def main() -> None:
    print(f"👀  Watching {WATCH_FOLDER} for new images …")
    handler = ImageDropHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_FOLDER, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
