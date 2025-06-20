import base64
import time
from datetime import datetime
from pathlib import Path

import winsound
from dotenv import load_dotenv
from openai import OpenAI
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------
WATCH_FOLDER = Path("D:/") / "Draw3DLine" / "Save"      # folder to watch for images
AUDIO_FOLDER = WATCH_FOLDER / "spoken"         # folder to keep MP3s  ← NEW
PROMPT_TEXT = (
    "You are a friendly assistant. "
    "Look at the image and give the user concise, constructive feedback."
)
MODEL_VISION = "gpt-4o-mini"
MODEL_TTS = "tts-1"
VOICE = "alloy"

# Make sure the audio folder exists  ← NEW
AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)

# Instantiate a single OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()


def talk(text: str) -> Path:
    """Generate speech with OpenAI TTS → WAV → play synchronously."""
    # ----- 1. request WAV instead of MP3 ------------------------------
    wav_bytes = client.audio.speech.create(
        model=MODEL_TTS,
        voice=VOICE,
        input=text,
        response_format="wav",
    ).content

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    wav_path = AUDIO_FOLDER / f"reply-{ts}.wav"
    wav_path.write_bytes(wav_bytes)

    winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
    print(f"🔊  Played and saved {wav_path}")
    return wav_path

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
