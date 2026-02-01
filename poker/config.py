import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
PETER_GRIFFIN_VOICE_ID = "d75c270eaee14c8aa1e9e980cc37cf1b"

# Camera settings
CAMERA_INDEX = 0
FRAME_INTERVAL = 0.5  # seconds between captures
