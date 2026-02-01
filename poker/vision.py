# import base64
# import json

# import cv2
# import numpy as np
# from openai import OpenAI
# from pydantic import BaseModel

# from poker.config import OPENROUTER_API_KEY
# from poker.models import Card, Rank, Suit


# def get_client() -> OpenAI:
#     """Get OpenAI client (lazy initialization)"""
#     return OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENROUTER_API_KEY,
#     )


# class TableReading(BaseModel):
#     """What Gemini sees on the table"""
#     dealer_cards: list[Card]
#     dealer_hole_hidden: bool
#     player_cards: list[Card]


# def capture_frame(camera_index: int = 0) -> np.ndarray:
#     """Capture a single frame from camera"""
#     cap = cv2.VideoCapture(camera_index)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         raise RuntimeError("Failed to capture frame")
#     return frame


# def frame_to_base64(frame: np.ndarray) -> str:
#     """Convert CV2 frame to base64 for API"""
#     _, buffer = cv2.imencode('.jpg', frame)
#     return base64.b64encode(buffer).decode('utf-8')


# def read_cards(frame: np.ndarray) -> TableReading:
#     """Use Gemini to identify cards on table"""
#     b64_image = frame_to_base64(frame)

#     client = get_client()
#     response = client.chat.completions.create(
#         model="google/gemini-2.0-flash-lite-001",
#         messages=[
#             {
#                 "role": "system",
#                 "content": """You are a card recognition system for blackjack.
# Analyze the image and identify all playing cards visible.

# The table layout:
# - DEALER cards are at the TOP of the image
# - PLAYER cards are at the BOTTOM of the image
# - Face-down cards appear as card backs (solid color/pattern)

# Respond in this exact JSON format:
# {
#     "dealer_cards": [{"rank": "A", "suit": "spades"}, ...],
#     "dealer_hole_hidden": true/false,
#     "player_cards": [{"rank": "10", "suit": "hearts"}, ...]
# }

# Ranks: 2,3,4,5,6,7,8,9,10,J,Q,K,A
# Suits: hearts, diamonds, clubs, spades

# If you see a face-down card in dealer's hand, set dealer_hole_hidden to true and only include the visible dealer card."""
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{b64_image}"
#                         }
#                     },
#                     {
#                         "type": "text",
#                         "text": "What cards are on the blackjack table? Dealer cards at top, player cards at bottom."
#                     }
#                 ]
#             }
#         ],
#         response_format={"type": "json_object"},
#         max_tokens=500,
#     )

#     data = json.loads(response.choices[0].message.content)

#     def parse_card(c: dict) -> Card:
#         return Card(
#             rank=Rank(c["rank"]),
#             suit=Suit(c["suit"])
#         )

#     return TableReading(
#         dealer_cards=[parse_card(c) for c in data.get("dealer_cards", [])],
#         dealer_hole_hidden=data.get("dealer_hole_hidden", True),
#         player_cards=[parse_card(c) for c in data.get("player_cards", [])]
#     )

from itertools import cycle
from typing import Optional

import numpy as np  # type: ignore
from pydantic import BaseModel

from poker.models import Card, Chip, ChipColor, ChipPosition, Rank, Suit


class TableReading(BaseModel):
    """What the (fake) vision system sees on the table"""
    dealer_cards: list[Card]
    dealer_hole_hidden: bool
    player_cards: list[Card]
    chips: list[Chip] = []


# Static chip layout for testing
TEST_CHIPS: list[Chip] = [
    Chip(color=ChipColor.PINK, position=ChipPosition.LEFT),
    Chip(color=ChipColor.PINK, position=ChipPosition.CENTER),
    Chip(color=ChipColor.PINK, position=ChipPosition.RIGHT),
    Chip(color=ChipColor.GREEN, position=ChipPosition.LEFT),
    Chip(color=ChipColor.GREEN, position=ChipPosition.CENTER),
    Chip(color=ChipColor.GREEN, position=ChipPosition.RIGHT),
    Chip(color=ChipColor.RED, position=ChipPosition.LEFT),
    Chip(color=ChipColor.RED, position=ChipPosition.CENTER),
    Chip(color=ChipColor.RED, position=ChipPosition.RIGHT),
    Chip(color=ChipColor.YELLOW, position=ChipPosition.LEFT),
    Chip(color=ChipColor.YELLOW, position=ChipPosition.CENTER),
    Chip(color=ChipColor.YELLOW, position=ChipPosition.RIGHT),
]

# Deterministic sequence of readings; advances each time read_cards is called.
_READINGS = cycle(
    [
        TableReading(
            dealer_cards=[],
            dealer_hole_hidden=True,
            player_cards=[],
            chips=TEST_CHIPS,
        ),
        TableReading(
            dealer_cards=[Card(rank=Rank.SIX, suit=Suit.SPADES)],
            dealer_hole_hidden=True,
            player_cards=[
                Card(rank=Rank.TEN, suit=Suit.HEARTS),
                Card(rank=Rank.SEVEN, suit=Suit.CLUBS),
            ],
            chips=TEST_CHIPS,
        ),
        TableReading(
            dealer_cards=[
                Card(rank=Rank.SIX, suit=Suit.SPADES),
                Card(rank=Rank.KING, suit=Suit.DIAMONDS),
            ],
            dealer_hole_hidden=False,
            player_cards=[
                Card(rank=Rank.TEN, suit=Suit.HEARTS),
                Card(rank=Rank.SEVEN, suit=Suit.CLUBS),
            ],
            chips=TEST_CHIPS,
        ),
    ]
)


def capture_frame(camera_index: int = 0) -> Optional[np.ndarray]:
    """
    Fake frame capture; returns None because downstream read_cards ignores it.
    """
    return None


def read_cards(frame: Optional[np.ndarray]) -> TableReading:
    """
    Return the next deterministic TableReading each time this is called.
    """
    reading = next(_READINGS)
    # Return a deep copy so callers cannot mutate the shared template.
    return reading.model_copy(deep=True)