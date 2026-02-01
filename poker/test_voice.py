"""Test Peter Griffin voice"""
import sys
from pathlib import Path

# Add parent dir to path so we can import poker
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.voice import speak

if __name__ == "__main__":
    speak("Holy crap Lois, I'm playing blackjack with a robot!")
    speak("Dealer's showing a 6? You're toast, buddy.")
