from dotenv import load_dotenv
import os

load_dotenv()

# CONSTANTS - REPLACE WITH YOUR API KEY
API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")


# Enhanced TTS models with emotional capabilities
TTS_MODELS = {
    "standard": "tts_models/en/ljspeech/glow-tts",        # Good balance
    "natural": "tts_models/en/vctk/vits",                 # More natural sounding
    "high_quality": "tts_models/en/vctk/fast_pitch",      # Higher quality
    "emotional": "tts_models/en/vctk/vits",               # Better for emotions
    "neural": "tts_models/en/ljspeech/tacotron2-DDC",     # Neural voice with more expression
}

# Speaker configs for emotional TTS
TTS_EMOTIONS = {
    "happy": {"speed": 1.15, "pitch": 1.1, "energy": 1.2},
    "sad": {"speed": 0.9, "pitch": 0.9, "energy": 0.8},
    "excited": {"speed": 1.2, "pitch": 1.15, "energy": 1.3},
    "calm": {"speed": 0.95, "pitch": 0.98, "energy": 0.9},
    "neutral": {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
}

# Whisper model options
WHISPER_MODELS = {
    "tiny": {"model": "tiny", "description": "Fastest, least accurate"},
    "base": {"model": "base", "description": "Good balance of speed and accuracy"},
    "small": {"model": "small", "description": "Better accuracy, slower"},
    "medium": {"model": "medium", "description": "High accuracy, but slower"},
    "large": {"model": "large", "description": "Best accuracy, slowest"}
}

# Parse command line arguments
tts_quality = "neural"  # Default to natural voice
whisper_size = "small"
language = "en"
