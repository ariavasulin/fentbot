#!/usr/bin/env python3
"""
Generate Peter Griffin zinger audio clips for the presentation.

Usage: python -m poker.generate_zingers

Output: Creates MP3 files in poker/zingers/
"""

import os
from pathlib import Path

from poker.config import FISH_AUDIO_API_KEY, PETER_GRIFFIN_VOICE_ID


ZINGERS = {
    "betting": "Alright, let's put fifty bucks on this one!",
    "blackjack": "BLACKJACK BABY! FREAKIN' SWEET!",
    "hit_risky": "This is more risky than that time I put metal in the microwave. HIT ME!",
    "stand_confident": "Twenty! I'm gonna rub this in Meg's stupid face! STAND baby!",
    "dealer_bust": "The dealer BUSTED! Holy crap, I actually won something!",
    "win": "I WON! The guys at the Clam are NOT gonna believe this!",
    "lose": "RIGGED! This whole casino is RIGGED I tell ya!",
    "lose_dramatic": "The house is rigged against me and I shall not abide by your rules. Good day sir!",
    "robot_dropped": "Oh come on! The stupid robot dropped the chip! This is worse than that time I tried to teach Chris how to drive!",
    "robot_failed": "Really? REALLY? The robot can't even put a chip down? Even Meg could do that!",
    "intro": "Alright, let's play some blackjack! I'm feeling lucky tonight.",
    "outro": "Alright, I'm out. Lois is gonna kill me when she sees the credit card bill.",
    "roadhouse": "Roadhouse.",
    "new_hand": "Alright, new hand! Let's go!",
    "push": "A push? That's like kissing your sister. Not that I would know anything about that.",
    "dysfunction": "I am presently afflicted by acute psychomotor dysfunction.",
    "drinking": "I should not have gone drinking this morning.",
}


def generate_audio(text: str, output_path: Path) -> None:
    """Generate audio file from text using Fish Audio TTS."""
    from fish_audio_sdk import Session, TTSRequest

    client = Session(FISH_AUDIO_API_KEY)
    audio_data = b""

    print(f"  Generating: {text[:50]}...")

    for chunk in client.tts(TTSRequest(
        text=text,
        reference_id=PETER_GRIFFIN_VOICE_ID
    )):
        audio_data += chunk

    with open(output_path, "wb") as f:
        f.write(audio_data)

    print(f"  Saved: {output_path}")


def main():
    output_dir = Path(__file__).parent / "zingers"
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {len(ZINGERS)} Peter Griffin zingers...")
    print(f"Output directory: {output_dir}\n")

    for name, text in ZINGERS.items():
        output_path = output_dir / f"{name}.mp3"
        generate_audio(text, output_path)

    print(f"\nDone! Generated {len(ZINGERS)} audio clips.")
    print(f"Files are in: {output_dir}")


if __name__ == "__main__":
    main()
