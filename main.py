import os
import json
import tempfile
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

# ── Auth ──────────────────────────────────────────────────────────────────────
SERVICE_SECRET = os.environ["SERVICE_SECRET"]  # set in Azure Container App env vars

# ── Azure Whisper config ──────────────────────────────────────────────────────
WHISPER_ENDPOINT   = os.environ["AZURE_OPENAI_WHISPER_ENDPOINT"].rstrip("/")
WHISPER_KEY        = os.environ["AZURE_OPENAI_WHISPER_KEY"]
WHISPER_DEPLOYMENT = os.environ.get("AZURE_OPENAI_WHISPER_DEPLOYMENT", "whisper")
WHISPER_API_VER    = os.environ.get("AZURE_OPENAI_WHISPER_API_VERSION", "2024-02-01")


class TranscriptRequest(BaseModel):
    videoId: str


def parse_json3(path: str) -> str:
    """Parse a yt-dlp json3 subtitle file into plain text."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            text = seg.get("utf8", "").replace("\n", " ").strip()
            if text:
                words.append(text)
    return " ".join(words).strip()


def extract_subtitles(video_id: str, tmp_dir: str) -> str:
    """Try to get auto-generated subtitles via yt-dlp (no audio download)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    result = subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--sub-langs", "en",
            "--sub-format", "json3",
            "-o", out_tmpl,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Find the downloaded json3 file
    for f in Path(tmp_dir).glob("*.json3"):
        transcript = parse_json3(str(f))
        if transcript:
            return transcript
    return ""


async def whisper_transcribe(audio_path: str) -> str:
    """Send audio file to Azure Whisper and return transcript text."""
    url = (
        f"{WHISPER_ENDPOINT}/openai/deployments/{WHISPER_DEPLOYMENT}"
        f"/audio/transcriptions?api-version={WHISPER_API_VER}"
    )
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            url,
            headers={"api-key": WHISPER_KEY},
            files={"file": (os.path.basename(audio_path), audio_bytes, "audio/mp4")},
            data={"model": WHISPER_DEPLOYMENT},
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Whisper error: {response.text}")
    return response.json().get("text", "")


def download_audio(video_id: str, tmp_dir: str) -> str:
    """Download audio-only stream via yt-dlp. Returns path to audio file."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_tmpl = os.path.join(tmp_dir, "audio.%(ext)s")

    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "--extract-audio",
            "--audio-format", "mp4",
            "--audio-quality", "5",   # lower quality = smaller file, faster
            "-o", out_tmpl,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )

    for ext in ("mp4", "m4a", "webm", "opus"):
        p = os.path.join(tmp_dir, f"audio.{ext}")
        if os.path.exists(p):
            return p
    raise HTTPException(status_code=500, detail="Audio download failed: file not found")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcript")
async def get_transcript(
    body: TranscriptRequest,
    x_service_secret: str = Header(...),
):
    if x_service_secret != SERVICE_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    video_id = body.videoId.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="videoId required")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # ── Step 1: Try subtitle extraction (fast, no audio download) ──────────
        try:
            transcript = extract_subtitles(video_id, tmp_dir)
            if transcript:
                return {"transcript": transcript, "source": "subtitles"}
        except Exception:
            pass  # fall through to Whisper

        # ── Step 2: Download audio → Whisper transcription ─────────────────────
        try:
            audio_path = download_audio(video_id, tmp_dir)
            transcript = await whisper_transcribe(audio_path)
            if transcript:
                return {"transcript": transcript, "source": "whisper"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return {"transcript": "", "source": "none"}
