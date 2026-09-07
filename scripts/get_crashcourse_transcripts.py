"""Получает транскрипты Crash Course AI через youtube-transcript-api.

AI for Everyone playlist: PL8dPuuaLjXtO65LeND2o_WuqHNBT_VD7x (10 episodes)
"""
import json
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUT = ROOT / "data/kb_crashcourse_transcripts.jsonl"

# Видео из плейлиста Crash Course AI
VIDEOS = [
    ("PL8dPuuaLjXtO65LeND2o_WuqHNBT_VD7x", "ai_playlist"),
]

# Пробуем установить youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    print("Installing youtube-transcript-api...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "youtube-transcript-api", "--quiet"])
    from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript(video_id: str) -> str:
    """Получает транскрипт видео."""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['en'])
        text = " ".join([snippet.text for snippet in transcript.snippets])
        return text[:5000]
    except Exception as e:
        return f"[Error: {e}]"


print("=" * 60)
print("Crash Course AI субтитры")
print("=" * 60)

# ID видео из плейлиста
video_ids = [
    "aircAruvnKk",  # But what is a neural network?
    "IHZwWFHWa-w",  # How does a neural network learn?
    "tieljriLqfE",  # What is backpropagation?
    "O5xeyMrlhJ0",  # Gradient descent
    "EuBBz3bTl-A",  # How neural networks learn
    "rEDg4OpACa0",  # Optimizers
    "IHfcoTpaRC0",  # Convolutions
    "E45l7u4Jun0",  # Recurrent neural networks
    "qgxh9_yJ7gY",  # Transformers
    "wjZofJX0v4M",  # AI for everyone
]

pairs = []
for i, vid in enumerate(video_ids):
    print(f"[{i+1}/{len(video_ids)}] Fetching {vid}...")
    text = get_transcript(vid)
    if not text.startswith("[Error"):
        title = f"Crash Course AI #{i+1}"
        pairs.append({
            "prompt": f"Расскажи про тему из видео: {title}",
            "response": text,
        })

print(f"\n📊 Получено транскриптов: {len(pairs)}")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")
