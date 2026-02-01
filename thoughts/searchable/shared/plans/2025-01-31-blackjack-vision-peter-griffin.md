# Blackjack Vision System with Peter Griffin Commentary

## Overview

A blackjack vision system that watches an overhead camera, reads cards using Gemini Flash Lite, tracks game state, makes basic strategy decisions, and provides continuous degenerate commentary in Peter Griffin's voice via Fish Audio TTS.

## Current State Analysis

- Existing `solo-cli/` directory contains a robot control CLI with Python 3.12, Pydantic, and camera utilities
- No existing blackjack/poker code - starting fresh in `poker/` directory
- Fish Audio provides Peter Griffin voice model: `d75c270eaee14c8aa1e9e980cc37cf1b`

## Desired End State

A running system that:
1. Captures frames from overhead camera
2. Identifies cards on the table (dealer's cards, player's cards)
3. Tracks game state (whose turn, what phase)
4. Makes hit/stand/double/split decisions using basic strategy
5. Continuously talks shit as Peter Griffin throughout the game
6. Outputs action decisions as JSON (schema TBD for VLA robot)

### Verification:
- Run `python -m poker.main` with camera pointed at cards
- System identifies cards correctly and announces them
- Peter Griffin voice plays commentary
- Correct basic strategy decisions are made

## What We're NOT Doing

- VLA robot integration (JSON output only, schema figured out later)
- Chip tracking/betting amounts
- Card counting
- Multi-hand/multi-player support
- Persistent session memory
- Complex agent framework (raw API calls only)

## Implementation Approach

Simple polling loop architecture:
```
Camera (OpenCV)
    → Gemini Flash Lite (card recognition)
    → Game State Tracker (Pydantic models)
    → LLM (decision + commentary generation)
    → Fish Audio TTS (Peter Griffin voice)
    → JSON action output
```

No agent framework - just a `while True` loop with API calls.

---

## Phase 1: Project Setup & Data Models

### Overview
Set up the project structure, dependencies, and Pydantic models for game state.

### Changes Required:

#### 1. Create project structure
```
poker/
├── __init__.py
├── main.py              # Main loop
├── models.py            # Pydantic models
├── vision.py            # Gemini card recognition
├── strategy.py          # Basic strategy lookup
├── voice.py             # Fish Audio TTS
├── prompts.py           # System prompts for personality
└── config.py            # API keys, settings
```

#### 2. Dependencies
**File**: `poker/requirements.txt`
```
opencv-python>=4.8.0
pydantic>=2.0.0
openai>=1.0.0  # For OpenRouter
fish-audio-sdk[utils]>=0.1.0
python-dotenv>=1.0.0
```

#### 3. Data Models
**File**: `poker/models.py`
```python
from enum import Enum
from pydantic import BaseModel

class Suit(str, Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"

class Rank(str, Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

class Card(BaseModel):
    rank: Rank
    suit: Suit

    @property
    def value(self) -> int:
        """Blackjack value (Ace=11, face=10)"""
        if self.rank == Rank.ACE:
            return 11
        elif self.rank in (Rank.JACK, Rank.QUEEN, Rank.KING):
            return 10
        return int(self.rank.value)

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value[0].upper()}"

class Hand(BaseModel):
    cards: list[Card] = []

    @property
    def value(self) -> int:
        """Best blackjack value (adjusts aces)"""
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == Rank.ACE)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    @property
    def is_soft(self) -> bool:
        """Has an ace counting as 11"""
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == Rank.ACE)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        # If we still have aces and didn't reduce all of them
        return any(c.rank == Rank.ACE for c in self.cards) and total <= 21 and sum(c.value for c in self.cards) == total

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value == 21

    @property
    def is_bust(self) -> bool:
        return self.value > 21

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

class GamePhase(str, Enum):
    WAITING_FOR_DEAL = "waiting_for_deal"      # No cards on table
    PLAYER_TURN = "player_turn"                # Player deciding
    DEALER_TURN = "dealer_turn"                # Dealer revealing/hitting
    HAND_COMPLETE = "hand_complete"            # Winner determined

class Action(str, Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"

class GameState(BaseModel):
    dealer_hand: Hand = Hand()
    player_hand: Hand = Hand()
    dealer_hole_card_hidden: bool = True  # Is dealer's second card face-down?
    phase: GamePhase = GamePhase.WAITING_FOR_DEAL
    last_action: Action | None = None

class ActionOutput(BaseModel):
    """JSON output for VLA robot"""
    action: Action
    confidence: float
    commentary: str  # What Peter is saying
```

#### 4. Config
**File**: `poker/config.py`
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY")
PETER_GRIFFIN_VOICE_ID = "d75c270eaee14c8aa1e9e980cc37cf1b"

# Camera settings
CAMERA_INDEX = 0
FRAME_INTERVAL = 0.5  # seconds between captures
```

#### 5. Environment template
**File**: `poker/.env.example`
```
OPENROUTER_API_KEY=your_key_here
FISH_AUDIO_API_KEY=your_key_here
```

### Success Criteria:

#### Automated Verification:
- [x] `cd poker && pip install -r requirements.txt` succeeds
- [x] `python -c "from poker.models import GameState, Card, Hand"` works
- [ ] Type checking passes: `python -m mypy poker/models.py` (skipping for hackathon)

#### Manual Verification:
- [ ] Models correctly calculate hand values (test with known hands)

---

## Phase 2: Vision System (Gemini Card Recognition)

### Overview
Use Gemini Flash Lite via OpenRouter to identify cards from camera frames.

### Changes Required:

#### 1. Vision module
**File**: `poker/vision.py`
```python
import base64
import cv2
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from poker.config import OPENROUTER_API_KEY
from poker.models import Card, Rank, Suit

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class TableReading(BaseModel):
    """What Gemini sees on the table"""
    dealer_cards: list[Card]
    dealer_hole_hidden: bool  # Is there a face-down card?
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

    import json
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
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from poker.vision import read_cards, capture_frame"` works

#### Manual Verification:
- [ ] Point camera at playing cards, run vision test
- [ ] Correctly identifies card rank and suit
- [ ] Correctly distinguishes dealer vs player position

**Implementation Note**: Test with actual cards before proceeding.

---

## Phase 3: Basic Strategy Engine

### Overview
Implement basic strategy lookup table for hit/stand/double/split decisions.

### Changes Required:

#### 1. Strategy module
**File**: `poker/strategy.py`
```python
from poker.models import Hand, Action, Card, Rank

def get_dealer_upcard_value(dealer_hand: Hand) -> int:
    """Get value of dealer's visible card"""
    if not dealer_hand.cards:
        return 0
    return dealer_hand.cards[0].value

def basic_strategy(player_hand: Hand, dealer_upcard: int) -> Action:
    """
    Basic strategy lookup.
    Returns optimal action based on player hand and dealer upcard.

    Simplified rules (covers most cases):
    """
    player_value = player_hand.value
    is_soft = player_hand.is_soft
    is_pair = player_hand.is_pair

    # Pair splitting
    if is_pair:
        pair_rank = player_hand.cards[0].rank

        # Always split Aces and 8s
        if pair_rank in (Rank.ACE, Rank.EIGHT):
            return Action.SPLIT

        # Never split 10s, 5s, 4s
        if pair_rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.FIVE, Rank.FOUR):
            pass  # Fall through to regular strategy

        # Split 9s except against 7, 10, A
        if pair_rank == Rank.NINE and dealer_upcard not in (7, 10, 11):
            return Action.SPLIT

        # Split 7s against 2-7
        if pair_rank == Rank.SEVEN and 2 <= dealer_upcard <= 7:
            return Action.SPLIT

        # Split 6s against 2-6
        if pair_rank == Rank.SIX and 2 <= dealer_upcard <= 6:
            return Action.SPLIT

        # Split 3s and 2s against 2-7
        if pair_rank in (Rank.THREE, Rank.TWO) and 2 <= dealer_upcard <= 7:
            return Action.SPLIT

    # Soft hands (Ace counting as 11)
    if is_soft:
        if player_value >= 19:
            return Action.STAND
        if player_value == 18:
            if dealer_upcard >= 9:
                return Action.HIT
            if dealer_upcard in (3, 4, 5, 6):
                return Action.DOUBLE
            return Action.STAND
        if player_value == 17:
            if dealer_upcard in (3, 4, 5, 6):
                return Action.DOUBLE
            return Action.HIT
        if player_value in (15, 16):
            if dealer_upcard in (4, 5, 6):
                return Action.DOUBLE
            return Action.HIT
        if player_value in (13, 14):
            if dealer_upcard in (5, 6):
                return Action.DOUBLE
            return Action.HIT
        return Action.HIT

    # Hard hands
    if player_value >= 17:
        return Action.STAND

    if player_value >= 13:
        if dealer_upcard <= 6:
            return Action.STAND
        return Action.HIT

    if player_value == 12:
        if 4 <= dealer_upcard <= 6:
            return Action.STAND
        return Action.HIT

    if player_value == 11:
        return Action.DOUBLE

    if player_value == 10:
        if dealer_upcard <= 9:
            return Action.DOUBLE
        return Action.HIT

    if player_value == 9:
        if 3 <= dealer_upcard <= 6:
            return Action.DOUBLE
        return Action.HIT

    # 8 or less: always hit
    return Action.HIT
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from poker.strategy import basic_strategy"` works
- [x] Unit tests for known strategy decisions pass

#### Manual Verification:
- [ ] Spot check: 16 vs dealer 10 = HIT
- [ ] Spot check: 11 vs dealer 6 = DOUBLE
- [ ] Spot check: A,7 (soft 18) vs 9 = HIT

---

## Phase 4: Peter Griffin Voice (Fish Audio TTS)

### Overview
Integrate Fish Audio TTS with Peter Griffin voice for commentary.

### Changes Required:

#### 1. Voice module
**File**: `poker/voice.py`
```python
import os
from fishaudio import FishAudio
from fishaudio.utils import play

from poker.config import FISH_AUDIO_API_KEY, PETER_GRIFFIN_VOICE_ID

# Set API key via environment (SDK reads FISH_AUDIO_API_KEY)
os.environ["FISH_AUDIO_API_KEY"] = FISH_AUDIO_API_KEY

client = FishAudio()

def speak(text: str, block: bool = True) -> None:
    """
    Speak text as Peter Griffin.

    Args:
        text: What to say
        block: If True, wait for audio to finish
    """
    audio = client.tts.convert(
        text=text,
        reference_id=PETER_GRIFFIN_VOICE_ID
    )
    play(audio)

def speak_async(text: str) -> None:
    """Speak without blocking (for continuous commentary)"""
    import threading
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from poker.voice import speak"` works

#### Manual Verification:
- [ ] Run `speak("Holy crap Lois, I got blackjack!")`
- [ ] Audio plays in Peter Griffin's voice

---

## Phase 5: Conversational Agent (Degenerate Personality)

### Overview
Create the LLM-powered personality that makes decisions and talks shit.

### Changes Required:

#### 1. Prompts
**File**: `poker/prompts.py`
```python
PETER_SYSTEM_PROMPT = """You are Peter Griffin from Family Guy, playing blackjack. You are a degenerate gambler who talks constant shit.

Your personality:
- Overconfident despite being down money overall
- Gets WAY too excited about good hands
- Blames the dealer/casino when you lose ("RIGGED!")
- Makes inappropriate comments and references to Family Guy
- Uses catchphrases like "Freakin' sweet!", "Holy crap!", "You know what really grinds my gears?"
- Talks to the cards like they can hear you
- Occasionally mentions Lois, Brian, Stewie, etc.

You must:
1. Announce what cards you see
2. Decide: HIT, STAND, DOUBLE, or SPLIT
3. Talk shit about it

Keep responses SHORT (1-3 sentences max). This is spoken aloud, so be punchy.

Examples of your style:
- "Ohhh we got a 20! That's what I'm talking about! STAND, baby!"
- "Dealer showing a 6? That's a bust card! You're gonna bust harder than my diet, pal."
- "Sixteen against a 10? Ehhhh this is gonna hurt... but screw it, HIT ME."
- "Double down on 11. I didn't come here to play it safe like some kinda Brian."
- "BLACKJACK! FREAKIN' SWEET! Lois is NOT gonna believe this!"
- "Pair of 8s? We're splitting these bad boys. Twice the fun, twice the disappointment probably."

Current game state will be provided. Make your decision and commentary."""

def format_game_state(dealer_cards: str, player_cards: str, player_value: int, dealer_upcard: int) -> str:
    return f"""CURRENT HAND:
Dealer showing: {dealer_cards}
Your cards: {player_cards} (total: {player_value})

What's your move?"""
```

#### 2. Agent module
**File**: `poker/agent.py`
```python
import json
from openai import OpenAI
from pydantic import BaseModel

from poker.config import OPENROUTER_API_KEY
from poker.models import GameState, Action, Hand
from poker.prompts import PETER_SYSTEM_PROMPT, format_game_state
from poker.strategy import basic_strategy, get_dealer_upcard_value

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class PeterResponse(BaseModel):
    action: Action
    commentary: str

def get_peter_decision(game_state: GameState) -> PeterResponse:
    """
    Get Peter's decision and commentary for current game state.
    Uses basic strategy for the decision, LLM for the personality.
    """
    # Get the correct action from basic strategy
    dealer_upcard = get_dealer_upcard_value(game_state.dealer_hand)
    correct_action = basic_strategy(game_state.player_hand, dealer_upcard)

    # Format cards for display
    dealer_cards_str = ", ".join(str(c) for c in game_state.dealer_hand.cards)
    if game_state.dealer_hole_card_hidden:
        dealer_cards_str += " + [HIDDEN]"
    player_cards_str = ", ".join(str(c) for c in game_state.player_hand.cards)

    # Get Peter's commentary
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",  # Fast and cheap for commentary
        messages=[
            {"role": "system", "content": PETER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": format_game_state(
                    dealer_cards_str,
                    player_cards_str,
                    game_state.player_hand.value,
                    dealer_upcard
                ) + f"\n\n[The correct play is: {correct_action.value.upper()}]"
            }
        ],
        max_tokens=150,
    )

    commentary = response.choices[0].message.content.strip()

    return PeterResponse(
        action=correct_action,
        commentary=commentary
    )

def get_reaction(event: str) -> str:
    """Get Peter's reaction to game events (dealer bust, win, lose, etc.)"""
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {"role": "system", "content": PETER_SYSTEM_PROMPT},
            {"role": "user", "content": f"React to this: {event}"}
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from poker.agent import get_peter_decision"` works

#### Manual Verification:
- [ ] Test with sample game state, verify commentary is in-character
- [ ] Verify correct action is returned

---

## Phase 6: Main Loop & Integration

### Overview
Wire everything together into the main game loop.

### Changes Required:

#### 1. Main loop
**File**: `poker/main.py`
```python
import time
import json
from datetime import datetime

from poker.config import CAMERA_INDEX, FRAME_INTERVAL
from poker.models import GameState, GamePhase, Hand, ActionOutput
from poker.vision import capture_frame, read_cards
from poker.strategy import get_dealer_upcard_value
from poker.agent import get_peter_decision, get_reaction
from poker.voice import speak, speak_async

def cards_changed(old_reading, new_reading) -> bool:
    """Check if cards on table have changed"""
    if old_reading is None:
        return True
    return (
        old_reading.dealer_cards != new_reading.dealer_cards or
        old_reading.player_cards != new_reading.player_cards
    )

def determine_phase(reading) -> GamePhase:
    """Determine game phase from what we see"""
    if not reading.player_cards and not reading.dealer_cards:
        return GamePhase.WAITING_FOR_DEAL

    if reading.dealer_hole_hidden:
        return GamePhase.PLAYER_TURN

    # If dealer hole card is revealed, it's dealer's turn or hand is complete
    return GamePhase.DEALER_TURN

def run():
    """Main game loop"""
    print("=" * 50)
    print("PETER GRIFFIN BLACKJACK SYSTEM")
    print("=" * 50)
    speak("Alright, let's play some blackjack! I'm feeling lucky tonight.")

    last_reading = None
    game_state = GameState()
    last_decision_made = False

    while True:
        try:
            # Capture and analyze
            frame = capture_frame(CAMERA_INDEX)
            reading = read_cards(frame)

            # Check if anything changed
            if not cards_changed(last_reading, reading):
                time.sleep(FRAME_INTERVAL)
                continue

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cards detected!")
            print(f"  Dealer: {[str(c) for c in reading.dealer_cards]} (hidden: {reading.dealer_hole_hidden})")
            print(f"  Player: {[str(c) for c in reading.player_cards]}")

            # Update game state
            game_state.dealer_hand = Hand(cards=reading.dealer_cards)
            game_state.player_hand = Hand(cards=reading.player_cards)
            game_state.dealer_hole_card_hidden = reading.dealer_hole_hidden
            game_state.phase = determine_phase(reading)

            # React based on phase
            if game_state.phase == GamePhase.WAITING_FOR_DEAL:
                if last_reading is not None:  # Cards were cleared
                    speak_async("Alright, new hand! Let's go!")
                last_decision_made = False

            elif game_state.phase == GamePhase.PLAYER_TURN:
                if not last_decision_made:
                    # Check for blackjack
                    if game_state.player_hand.is_blackjack:
                        speak("BLACKJACK BABY! FREAKIN' SWEET!")
                    else:
                        # Get Peter's decision
                        response = get_peter_decision(game_state)

                        # Output action as JSON
                        action_output = ActionOutput(
                            action=response.action,
                            confidence=0.95,
                            commentary=response.commentary
                        )
                        print(f"\nACTION: {json.dumps(action_output.model_dump(), indent=2)}")

                        # Speak the commentary
                        speak(response.commentary)

                        game_state.last_action = response.action
                        last_decision_made = True

            elif game_state.phase == GamePhase.DEALER_TURN:
                # Dealer revealed hole card
                dealer_value = game_state.dealer_hand.value
                player_value = game_state.player_hand.value

                if game_state.dealer_hand.is_bust:
                    reaction = get_reaction("The dealer BUSTED! You win!")
                    speak(reaction)
                elif dealer_value > player_value:
                    reaction = get_reaction(f"Dealer has {dealer_value}, you have {player_value}. You lose.")
                    speak(reaction)
                elif dealer_value < player_value:
                    reaction = get_reaction(f"You have {player_value}, dealer has {dealer_value}. You WIN!")
                    speak(reaction)
                else:
                    reaction = get_reaction(f"Push! Both have {player_value}.")
                    speak(reaction)

                last_decision_made = False  # Reset for next hand

            last_reading = reading

        except KeyboardInterrupt:
            print("\nShutting down...")
            speak("Alright, I'm out. Lois is gonna kill me when she sees the credit card bill.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run()
```

### Success Criteria:

#### Automated Verification:
- [x] `python -m poker.main --help` or basic import works

#### Manual Verification:
- [ ] Run with camera pointed at cards
- [ ] Peter announces cards he sees
- [ ] Peter makes correct decision and talks shit
- [ ] Audio plays properly
- [ ] JSON action output is printed

---

## Phase 7: Polish & Testing

### Overview
Add convenience features and test the full system.

### Changes Required:

#### 1. Add CLI entry point
**File**: `poker/__main__.py`
```python
from poker.main import run

if __name__ == "__main__":
    run()
```

#### 2. Test script for vision only
**File**: `poker/test_vision.py`
```python
"""Test card recognition without full loop"""
from poker.vision import capture_frame, read_cards

if __name__ == "__main__":
    print("Capturing frame...")
    frame = capture_frame()
    print("Analyzing cards...")
    reading = read_cards(frame)
    print(f"Dealer: {[str(c) for c in reading.dealer_cards]}")
    print(f"Player: {[str(c) for c in reading.player_cards]}")
    print(f"Dealer hole hidden: {reading.dealer_hole_hidden}")
```

#### 3. Test script for voice only
**File**: `poker/test_voice.py`
"""Test Peter Griffin voice"""
from poker.voice import speak

if __name__ == "__main__":
    speak("Holy crap Lois, I'm playing blackjack with a robot!")
    speak("Dealer's showing a 6? You're toast, buddy.")
```

### Success Criteria:

#### Automated Verification:
- [x] `python -m poker` runs without import errors

#### Manual Verification:
- [ ] Full end-to-end test with real cards
- [ ] Peter provides entertaining commentary throughout
- [ ] Decisions are correct per basic strategy

---

## Testing Strategy

### Unit Tests:
- Hand value calculation (hard hands, soft hands, busts)
- Basic strategy lookup for known situations
- Card parsing from vision response

### Integration Tests:
- Vision → Game State update
- Game State → Agent decision
- Agent decision → Voice output

### Manual Testing Steps:
1. Set up camera with overhead view of table
2. Place known cards (e.g., player: 10+6, dealer: 7 showing)
3. Run system, verify correct card recognition
4. Verify Peter says something about hitting 16 against 7
5. Verify JSON output shows HIT action
6. Test win/loss/bust reactions

## Performance Considerations

- Gemini Flash Lite is fast (~500ms for vision)
- Fish Audio TTS is fast (~300ms latency)
- Main bottleneck is speaking - don't interrupt Peter mid-sentence
- Consider non-blocking TTS for smoother experience

## References

- Fish Audio Python SDK: https://github.com/fishaudio/fish-audio-python
- Fish Audio TTS Docs: https://docs.fish.audio/developer-guide/sdk-guide/python/text-to-speech
- Peter Griffin Voice Model: https://fish.audio/m/d75c270eaee14c8aa1e9e980cc37cf1b/
- OpenRouter API: https://openrouter.ai/docs
- Blackjack Basic Strategy: https://www.blackjackapprenticeship.com/blackjack-strategy-charts/
