"""Test card recognition without full loop"""
import sys
from pathlib import Path

# Add parent dir to path so we can import poker
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.vision import capture_frame, read_cards

if __name__ == "__main__":
    print("Capturing frame...")
    frame = capture_frame()
    print("Analyzing cards...")
    reading = read_cards(frame)
    print(f"Dealer: {[str(c) for c in reading.dealer_cards]}")
    print(f"Player: {[str(c) for c in reading.player_cards]}")
    print(f"Dealer hole hidden: {reading.dealer_hole_hidden}")
