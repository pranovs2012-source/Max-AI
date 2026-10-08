"""Max's voice: text-to-speech with ElevenLabs.

Set ELEVENLABS_API_KEY (and optionally ELEVENLABS_VOICE_ID / ELEVENLABS_MODEL). The default voice is
ElevenLabs' premade "Daniel" — a calm, precise British voice that suits an AI butler. Any voice from
your ElevenLabs Voice Library works: copy its voice ID into ELEVENLABS_VOICE_ID.
Without a key the chat page falls back to the browser's built-in speech.
"""
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections import OrderedDict

DEFAULT_VOICE = "onwK4e9ZLuTAKqWW03F9"   # ElevenLabs premade voice "Daniel"
DEFAULT_MODEL = "eleven_flash_v2_5"      # fast, multilingual
MAX_CHARS = 900                          # keep replies short to save credits
_cache = OrderedDict()


class VoiceError(Exception):
    pass


def enabled():
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def speakable(text):
    """What Max says out loud: no code blocks, links or markdown, cut at a sentence end."""
    text = re.sub(r"```.*?(```|$)", " I've put the code on your screen. ", text, flags=re.S)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[`*_#>|]", "", text)
    text = re.sub(r"^\s*(\d+)\.\s+", r"\1: ", text, flags=re.M)   # "1. Bugatti" -> "1: Bugatti"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:end + 1] if end > 200 else cut
    return text


def synthesize(text):
    """MP3 bytes for the text."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise VoiceError("ELEVENLABS_API_KEY is not set")
    text = speakable(text)
    if not text:
        raise VoiceError("Nothing to say")
    voice = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)
    model = os.environ.get("ELEVENLABS_MODEL", DEFAULT_MODEL)
    cache_key = hashlib.sha256(f"{voice}|{model}|{text}".encode()).hexdigest()
    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        return _cache[cache_key]
    body = json.dumps({
        "text": text,
        "model_id": model,
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.8, "style": 0.15, "use_speaker_boost": True},
    }).encode()
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
        data=body, method="POST",
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            audio = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise VoiceError(f"ElevenLabs returned {e.code}: {detail}") from e
    except Exception as e:
        raise VoiceError(f"Could not reach ElevenLabs: {e}") from e
    _cache[cache_key] = audio
    if len(_cache) > 40:
        _cache.popitem(last=False)
    return audio
