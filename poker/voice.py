import io
import os
import threading

from poker.config import FISH_AUDIO_API_KEY, PETER_GRIFFIN_VOICE_ID


def get_client():
    """Get Fish Audio client (lazy initialization)"""
    from fish_audio_sdk import Session
    return Session(FISH_AUDIO_API_KEY)


# def speak(text: str, block: bool = True) -> None:
#     """
#     Speak text as Peter Griffin.

#     Args:
#         text: What to say
#         block: If True, wait for audio to finish
#     """
#     import sounddevice as sd
#     import soundfile as sf
#     from fish_audio_sdk import TTSRequest

#     client = get_client()
#     audio_data = b""

#     for chunk in client.tts(TTSRequest(
#         text=text,
#         reference_id=PETER_GRIFFIN_VOICE_ID
#     )):
#         audio_data += chunk

#     # Play the audio using sounddevice
#     audio_io = io.BytesIO(audio_data)
#     data, samplerate = sf.read(audio_io)
#     sd.play(data, samplerate)
#     if block:
#         sd.wait()


# def speak_async(text: str) -> None:
#     """Speak without blocking (for continuous commentary)"""
#     thread = threading.Thread(target=speak, args=(text,))
#     thread.daemon = True
#     thread.start()

def speak(text: str, block: bool = True) -> None:
    print(f"SPEAKING: {text}")

def speak_async(text: str) -> None:
    print(f"SPEAKING ASYNC: {text}")
