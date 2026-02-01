import base64
import json

import cv2
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from poker.config import OPENROUTER_API_KEY
from poker.models import Card, Rank, Suit


def get_client() -> OpenAI:
    """Get OpenAI client (lazy initialization)"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


class TableReading(BaseModel):
    """What Gemini sees on the table"""
    dealer_cards: list[Card]
    dealer_hole_hidden: bool
    player_cards: list[Card]


def capture_frame(camera_index: int = 0) -> np.ndarray:
    """Capture a single frame from camera"""
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert CV2 frame to base64 for API"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def read_cards(frame: np.ndarray) -> TableReading:
    """Use Gemini to identify cards on table"""
    b64_image = frame_to_base64(frame)

    client = get_client()
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "system",
                "content": """You are a card recognition system for blackjack.
Analyze the image and identify all playing cards visible.

The table layout:
- DEALER cards are at the TOP of the image
- PLAYER cards are at the BOTTOM of the image
- Face-down cards appear as card backs (solid color/pattern)

Respond in this exact JSON format:
{
    "dealer_cards": [{"rank": "A", "suit": "spades"}, ...],
    "dealer_hole_hidden": true/false,
    "player_cards": [{"rank": "10", "suit": "hearts"}, ...]
}

Ranks: 2,3,4,5,6,7,8,9,10,J,Q,K,A
Suits: hearts, diamonds, clubs, spades

If you see a face-down card in dealer's hand, set dealer_hole_hidden to true and only include the visible dealer card."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What cards are on the blackjack table? Dealer cards at top, player cards at bottom."
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    data = json.loads(response.choices[0].message.content)

    def parse_card(c: dict) -> Card:
        return Card(
            rank=Rank(c["rank"]),
            suit=Suit(c["suit"])
        )

    return TableReading(
        dealer_cards=[parse_card(c) for c in data.get("dealer_cards", [])],
        dealer_hole_hidden=data.get("dealer_hole_hidden", True),
        player_cards=[parse_card(c) for c in data.get("player_cards", [])]
    )
