# Poker Vision Tracker - Implementation Plan

## Overview

A camera-based Texas Hold'em game state tracker that:
1. Watches the table from overhead camera
2. Uses Gemini Flash Lite (via OpenRouter) to analyze frames every 5 seconds
3. Tracks community cards, pot, player bets, and turns
4. Outputs chip movement actions for the VLA when it's our turn

**Integration Point**: This system outputs `ChipMove` commands that map to `move_chip(from_pos, to_pos)` calls in the existing chip manipulation system (see `docs/CHIP_MANIPULATION_PLAN.md`).

---

## Physical Setup

```
Camera POV (overhead tripod, looking down at table)
┌─────────────────────────────────────────────────────┐
│                                                     │
│     P3 chips [1][5][10][50]                        │
│                                                     │
│  P2 chips                      P4 chips            │
│  [1][5][10][50]    ┌─────┐    [1][5][10][50]       │
│                    │ POT │                         │
│                    └─────┘                         │
│                  [  ] [  ] [  ] [  ] [  ]          │
│                   Community Cards                   │
│                                                     │
│     P1 (US) chips [1][5][10][50]                   │
│     [Hole Cards visible to camera]                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Setup Rules:**
- Each player's chips arranged L→R: 1s, 5s, 10s, 50s (stacked by denomination)
- Community cards dealt in clear horizontal row
- Pot is distinct central zone
- Our hole cards face-up/visible to camera
- Dealer button visible for tracking

---

## Tech Stack

- **Python 3.12** - Runtime (use existing solo-cli venv)
- **Pydantic** - Data models
- **OpenRouter API** - LLM gateway
- **Gemini Flash Lite** - Vision model (`google/gemini-2.0-flash-lite-001`)
- **OpenCV** - Camera capture
- **asyncio** - Main loop timing

---

## File Structure

```
poker/
├── __init__.py
├── main.py              # Entry point, CLI args, main loop
├── models.py            # Pydantic models (GameState, Card, etc.)
├── vision.py            # Camera capture + Gemini API calls
├── tracker.py           # State machine, applies vision deltas
├── strategy.py          # Simple decision algorithm
├── actions.py           # Convert decisions to ChipMove commands
└── config.py            # OpenRouter API key, camera index, etc.
```

---

## Phase 1: Core Models

### File: `poker/models.py`

```python
from pydantic import BaseModel, computed_field
from enum import Enum
from typing import Optional

class Suit(str, Enum):
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"

class Card(BaseModel):
    rank: str  # "2"-"10", "J", "Q", "K", "A"
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.value}"

class GamePhase(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

class PlayerAction(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

class ChipStack(BaseModel):
    ones: int = 0
    fives: int = 0
    tens: int = 0
    fifties: int = 0

    @computed_field
    @property
    def total(self) -> int:
        return self.ones + (self.fives * 5) + (self.tens * 10) + (self.fifties * 50)

    def can_afford(self, amount: int) -> bool:
        return self.total >= amount

class Player(BaseModel):
    seat: int
    chips: ChipStack
    current_bet: int = 0  # chips committed this betting round
    is_folded: bool = False
    is_us: bool = False

class GameState(BaseModel):
    # Setup (from CLI args)
    num_players: int
    our_seat: int
    hole_cards: tuple[Card, Card]
    big_blind: int = 10

    # Tracked state
    phase: GamePhase = GamePhase.PREFLOP
    community_cards: list[Card] = []
    pot: int = 0
    current_bet_to_call: int = 0
    players: list[Player] = []
    action_on_seat: int = 0
    dealer_seat: int = 0

    @computed_field
    @property
    def is_our_turn(self) -> bool:
        return self.action_on_seat == self.our_seat

    @computed_field
    @property
    def our_player(self) -> Player:
        return next(p for p in self.players if p.is_us)

    @computed_field
    @property
    def amount_to_call(self) -> int:
        return self.current_bet_to_call - self.our_player.current_bet

# Vision model response
class VisionDelta(BaseModel):
    """What changed between frames"""
    new_community_cards: list[Card] = []
    player_actions: list[dict] = []  # [{"seat": 1, "action": "bet", "amount": 10}]
    pot_change: int = 0
    action_now_on_seat: Optional[int] = None
    phase_change: Optional[GamePhase] = None

# VLA output
class ChipMove(BaseModel):
    """Single chip movement instruction for VLA"""
    chip_value: int  # 1, 5, 10, or 50
    count: int
    from_location: str  # "our_ones", "our_fives", etc.
    to_location: str    # "pot"

class ActionPlan(BaseModel):
    """What we tell the VLA to do"""
    action_type: PlayerAction
    total_amount: int = 0
    chip_moves: list[ChipMove] = []
```

### Success Criteria
- [ ] All models defined with proper types
- [ ] ChipStack.total computed correctly
- [ ] GameState.is_our_turn works
- [ ] Models serialize/deserialize to JSON

---

## Phase 2: Vision System

### File: `poker/vision.py`

```python
import cv2
import base64
import httpx
from .models import VisionDelta, GameState

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def capture_frame(camera_index: int = 0) -> bytes:
    """Capture a single frame from camera, return as JPEG bytes."""
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def frame_to_base64(frame_bytes: bytes) -> str:
    """Convert frame bytes to base64 string."""
    return base64.b64encode(frame_bytes).decode('utf-8')

async def analyze_frame(
    frame_bytes: bytes,
    current_state: GameState,
    api_key: str
) -> VisionDelta:
    """Send frame to Gemini Flash Lite, get back what changed."""

    prompt = build_analysis_prompt(current_state)
    base64_image = frame_to_base64(frame_bytes)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-2.0-flash-lite-001",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            },
            timeout=30.0
        )

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return VisionDelta.model_validate_json(content)

def build_analysis_prompt(state: GameState) -> str:
    """Build the prompt for frame analysis."""

    players_desc = "\n".join([
        f"  - Seat {p.seat}: {p.chips.total} chips, bet {p.current_bet}, {'FOLDED' if p.is_folded else 'active'}"
        for p in state.players
    ])

    community = ", ".join(str(c) for c in state.community_cards) or "none"

    return f"""You are tracking a Texas Hold'em poker game from an overhead camera.

CURRENT KNOWN STATE:
- Phase: {state.phase.value}
- Community cards: {community}
- Pot: {state.pot}
- Current bet to call: {state.current_bet_to_call}
- Action on: Seat {state.action_on_seat}
- Players:
{players_desc}

CHIP LAYOUT: Each player has 4 stacks left-to-right: 1s, 5s, 10s, 50s

Analyze this frame and report ONLY what has CHANGED since the known state:

1. New community cards dealt? (flop = 3 cards, turn = 1, river = 1)
2. Any player bet/raised? (chips moved from their stack toward pot)
3. Any player folded? (cards pushed away/face down)
4. Phase change? (preflop -> flop -> turn -> river -> showdown)
5. Whose turn is it now?

Respond with valid JSON matching this schema:
{{
  "new_community_cards": [  // only NEW cards, not all cards
    {{"rank": "A", "suit": "h"}}  // suit: h/d/c/s
  ],
  "player_actions": [  // actions since last frame
    {{"seat": 1, "action": "bet", "amount": 20}}
  ],
  "pot_change": 0,  // how much pot increased
  "action_now_on_seat": 2,  // whose turn now (null if unclear)
  "phase_change": null  // "flop", "turn", "river", "showdown", or null
}}

If nothing changed, return empty arrays and nulls. Be conservative - only report what you can clearly see."""
```

### Success Criteria
- [ ] Camera capture works (test with `capture_frame()`)
- [ ] Base64 encoding works
- [ ] OpenRouter API call succeeds
- [ ] Response parses to VisionDelta

---

## Phase 3: State Tracker

### File: `poker/tracker.py`

```python
from .models import GameState, VisionDelta, GamePhase, Player, ChipStack, Card

class GameTracker:
    """Maintains game state, applies vision deltas."""

    def __init__(self, initial_state: GameState):
        self.state = initial_state
        self.history: list[VisionDelta] = []

    def apply_delta(self, delta: VisionDelta) -> None:
        """Apply vision delta to current state."""

        # Add new community cards
        for card in delta.new_community_cards:
            if card not in self.state.community_cards:
                self.state.community_cards.append(card)

        # Process player actions
        for action in delta.player_actions:
            seat = action["seat"]
            player = next(p for p in self.state.players if p.seat == seat)

            if action["action"] == "fold":
                player.is_folded = True
            elif action["action"] in ("bet", "raise", "call", "all_in"):
                amount = action.get("amount", 0)
                player.current_bet += amount
                self.state.pot += amount
                if amount > self.state.current_bet_to_call:
                    self.state.current_bet_to_call = amount

        # Update pot
        if delta.pot_change:
            self.state.pot += delta.pot_change

        # Update action
        if delta.action_now_on_seat is not None:
            self.state.action_on_seat = delta.action_now_on_seat

        # Phase change
        if delta.phase_change:
            self.state.phase = delta.phase_change
            # Reset current bets on new phase
            for p in self.state.players:
                p.current_bet = 0
            self.state.current_bet_to_call = 0

        self.history.append(delta)

    def infer_phase_from_cards(self) -> GamePhase:
        """Infer phase from community card count."""
        n = len(self.state.community_cards)
        if n == 0:
            return GamePhase.PREFLOP
        elif n == 3:
            return GamePhase.FLOP
        elif n == 4:
            return GamePhase.TURN
        elif n == 5:
            return GamePhase.RIVER
        return self.state.phase
```

### Success Criteria
- [ ] Tracker correctly applies fold actions
- [ ] Tracker correctly applies bet/raise amounts
- [ ] Phase transitions work
- [ ] Current bet resets on new phase

---

## Phase 4: Strategy (Simple)

### File: `poker/strategy.py`

```python
from .models import GameState, PlayerAction, Card

# Preflop hand rankings (simplified)
PREMIUM_HANDS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
STRONG_HANDS = {"TT", "99", "AQs", "AQo", "AJs", "KQs"}
PLAYABLE_HANDS = {"88", "77", "66", "ATs", "KJs", "QJs", "JTs"}

def get_hand_key(hole_cards: tuple[Card, Card]) -> str:
    """Convert hole cards to hand key like 'AKs' or 'QJo'."""
    c1, c2 = hole_cards

    # Order by rank
    rank_order = "23456789TJQKA"
    if rank_order.index(c1.rank) < rank_order.index(c2.rank):
        c1, c2 = c2, c1

    suited = "s" if c1.suit == c2.suit else "o"

    if c1.rank == c2.rank:
        return f"{c1.rank}{c2.rank}"  # Pairs like "AA"
    else:
        return f"{c1.rank}{c2.rank}{suited}"

def decide_action(state: GameState) -> tuple[PlayerAction, int]:
    """
    Simple poker strategy. Returns (action, amount).

    Preflop: Play tight (premium/strong hands only)
    Postflop: Check if free, call small bets, fold to big bets
    """

    hand_key = get_hand_key(state.hole_cards)
    amount_to_call = state.amount_to_call
    pot = state.pot
    our_stack = state.our_player.chips.total

    # === PREFLOP ===
    if state.phase == "preflop":
        if hand_key in PREMIUM_HANDS:
            # Raise 3x BB with premium hands
            raise_amount = state.big_blind * 3
            return (PlayerAction.RAISE, raise_amount)

        elif hand_key in STRONG_HANDS:
            if amount_to_call <= state.big_blind * 2:
                return (PlayerAction.CALL, amount_to_call)
            else:
                return (PlayerAction.FOLD, 0)

        elif hand_key in PLAYABLE_HANDS:
            if amount_to_call <= state.big_blind:
                return (PlayerAction.CALL, amount_to_call)
            else:
                return (PlayerAction.FOLD, 0)

        else:
            # Junk hand
            if amount_to_call == 0:
                return (PlayerAction.CHECK, 0)
            else:
                return (PlayerAction.FOLD, 0)

    # === POSTFLOP (very simple) ===
    else:
        # If we can check, check
        if amount_to_call == 0:
            return (PlayerAction.CHECK, 0)

        # Call if bet is less than 1/3 pot
        if amount_to_call <= pot // 3:
            return (PlayerAction.CALL, amount_to_call)

        # Fold to big bets (for now)
        return (PlayerAction.FOLD, 0)
```

### Success Criteria
- [ ] Hand key generation works (AA, AKs, AKo, etc.)
- [ ] Premium hands raise
- [ ] Junk hands fold to bets
- [ ] Postflop checks when possible

---

## Phase 5: Action Generation

### File: `poker/actions.py`

```python
from .models import ActionPlan, ChipMove, PlayerAction, ChipStack

def make_change(amount: int, stack: ChipStack) -> list[ChipMove]:
    """
    Figure out which chips to use for a given amount.
    Greedy: use largest denominations first.
    Returns list of ChipMove instructions.
    """
    moves = []
    remaining = amount

    # Try 50s first
    if remaining >= 50 and stack.fifties > 0:
        count = min(remaining // 50, stack.fifties)
        moves.append(ChipMove(
            chip_value=50,
            count=count,
            from_location="our_fifties",
            to_location="pot"
        ))
        remaining -= count * 50

    # Then 10s
    if remaining >= 10 and stack.tens > 0:
        count = min(remaining // 10, stack.tens)
        moves.append(ChipMove(
            chip_value=10,
            count=count,
            from_location="our_tens",
            to_location="pot"
        ))
        remaining -= count * 10

    # Then 5s
    if remaining >= 5 and stack.fives > 0:
        count = min(remaining // 5, stack.fives)
        moves.append(ChipMove(
            chip_value=5,
            count=count,
            from_location="our_fives",
            to_location="pot"
        ))
        remaining -= count * 5

    # Then 1s
    if remaining >= 1 and stack.ones > 0:
        count = min(remaining, stack.ones)
        moves.append(ChipMove(
            chip_value=1,
            count=count,
            from_location="our_ones",
            to_location="pot"
        ))
        remaining -= count

    if remaining > 0:
        raise ValueError(f"Cannot make exact change for {amount}, {remaining} remaining")

    return moves

def generate_action_plan(action: PlayerAction, amount: int, stack: ChipStack) -> ActionPlan:
    """Convert a decision into chip movements."""

    if action in (PlayerAction.FOLD, PlayerAction.CHECK):
        return ActionPlan(action_type=action, total_amount=0, chip_moves=[])

    if action in (PlayerAction.CALL, PlayerAction.BET, PlayerAction.RAISE):
        chip_moves = make_change(amount, stack)
        return ActionPlan(
            action_type=action,
            total_amount=amount,
            chip_moves=chip_moves
        )

    return ActionPlan(action_type=action)
```

### Success Criteria
- [ ] make_change correctly decomposes amounts
- [ ] Uses largest denominations first
- [ ] Raises error if can't make exact change
- [ ] Fold/check return empty chip_moves

---

## Phase 6: Main Loop

### File: `poker/main.py`

```python
import asyncio
import argparse
from .models import GameState, Player, ChipStack, Card, Suit, GamePhase
from .vision import capture_frame, analyze_frame
from .tracker import GameTracker
from .strategy import decide_action
from .actions import generate_action_plan
from .config import OPENROUTER_API_KEY, CAMERA_INDEX

def parse_card(s: str) -> Card:
    """Parse card string like 'Ah' or '10s'."""
    suit_char = s[-1].lower()
    rank = s[:-1].upper()
    if rank == "10":
        rank = "T"  # Normalize
    suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return Card(rank=rank, suit=suit_map[suit_char])

def create_initial_state(args) -> GameState:
    """Create initial game state from CLI args."""

    # Parse hole cards
    hole_cards = (parse_card(args.card1), parse_card(args.card2))

    # Create players
    players = []
    for i in range(args.num_players):
        seat = i + 1
        players.append(Player(
            seat=seat,
            chips=ChipStack(
                ones=args.ones,
                fives=args.fives,
                tens=args.tens,
                fifties=args.fifties
            ),
            is_us=(seat == args.our_seat)
        ))

    # Dealer is seat 1 by default, action starts left of BB
    dealer_seat = 1
    bb_seat = (dealer_seat % args.num_players) + 1
    action_seat = (bb_seat % args.num_players) + 1

    return GameState(
        num_players=args.num_players,
        our_seat=args.our_seat,
        hole_cards=hole_cards,
        big_blind=args.big_blind,
        players=players,
        dealer_seat=dealer_seat,
        action_on_seat=action_seat,
        phase=GamePhase.PREFLOP
    )

async def main_loop(tracker: GameTracker, interval: float = 5.0):
    """Main vision loop."""

    print(f"Starting poker tracker. Polling every {interval}s...")
    print(f"Our hole cards: {tracker.state.hole_cards[0]}, {tracker.state.hole_cards[1]}")
    print(f"Watching for our turn (seat {tracker.state.our_seat})...")

    while True:
        try:
            # Capture frame
            frame = capture_frame(CAMERA_INDEX)

            # Analyze with vision
            delta = await analyze_frame(frame, tracker.state, OPENROUTER_API_KEY)

            # Apply changes
            if delta.new_community_cards or delta.player_actions or delta.phase_change:
                print(f"\n[UPDATE] {delta.model_dump_json(indent=2)}")
                tracker.apply_delta(delta)

            # Check if our turn
            if tracker.state.is_our_turn:
                print("\n*** IT'S OUR TURN ***")

                # Decide action
                action, amount = decide_action(tracker.state)
                print(f"Decision: {action.value} {amount}")

                # Generate chip moves
                plan = generate_action_plan(
                    action,
                    amount,
                    tracker.state.our_player.chips
                )

                print(f"Action plan: {plan.model_dump_json(indent=2)}")

                # TODO: Send to VLA
                # For now, just print and wait for next frame to see result

            await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Poker Vision Tracker")
    parser.add_argument("--num-players", type=int, required=True, help="Number of players")
    parser.add_argument("--our-seat", type=int, required=True, help="Our seat number (1-indexed)")
    parser.add_argument("--card1", type=str, required=True, help="First hole card (e.g., 'Ah')")
    parser.add_argument("--card2", type=str, required=True, help="Second hole card (e.g., 'Kd')")
    parser.add_argument("--ones", type=int, default=10, help="Starting 1-chips per player")
    parser.add_argument("--fives", type=int, default=10, help="Starting 5-chips per player")
    parser.add_argument("--tens", type=int, default=10, help="Starting 10-chips per player")
    parser.add_argument("--fifties", type=int, default=4, help="Starting 50-chips per player")
    parser.add_argument("--big-blind", type=int, default=10, help="Big blind amount")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")

    args = parser.parse_args()

    initial_state = create_initial_state(args)
    tracker = GameTracker(initial_state)

    asyncio.run(main_loop(tracker, args.interval))

if __name__ == "__main__":
    main()
```

### File: `poker/config.py`

```python
import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CAMERA_INDEX = int(os.environ.get("POKER_CAMERA_INDEX", "0"))
```

### Success Criteria
- [ ] CLI parses all arguments
- [ ] Initial state created correctly
- [ ] Main loop runs every 5 seconds
- [ ] Vision delta applied
- [ ] Decision made when our turn
- [ ] Action plan generated

---

## Usage

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-..."

# Run tracker
python -m poker.main \
    --num-players 4 \
    --our-seat 1 \
    --card1 "Ah" \
    --card2 "Kd" \
    --ones 10 \
    --fives 10 \
    --tens 10 \
    --fifties 4 \
    --big-blind 10 \
    --interval 5
```

---

## VLA Integration (Future)

The `ActionPlan.chip_moves` list maps directly to `move_chip()` calls:

```python
# In future integration:
for move in plan.chip_moves:
    for _ in range(move.count):
        # Map locations to grid positions
        from_pos = CHIP_LOCATIONS[move.from_location]
        to_pos = CHIP_LOCATIONS["pot"]
        move_chip(robot, grid_cal, pick_policy, drop_policy, from_pos, to_pos)
```

Grid position mapping (to be calibrated):
```python
CHIP_LOCATIONS = {
    "our_ones": (7, 0),    # Bottom-left of our chip area
    "our_fives": (7, 1),
    "our_tens": (7, 2),
    "our_fifties": (7, 3),
    "pot": (3, 4),         # Center of table
}
```

---

## Implementation Order

1. **Phase 1**: Models (`models.py`) - 15 min
2. **Phase 2**: Vision (`vision.py`, `config.py`) - 30 min
3. **Phase 3**: Tracker (`tracker.py`) - 20 min
4. **Phase 4**: Strategy (`strategy.py`) - 20 min
5. **Phase 5**: Actions (`actions.py`) - 15 min
6. **Phase 6**: Main loop (`main.py`) - 30 min

**Total: ~2 hours**

---

## Dependencies

Add to existing venv or create new:

```bash
pip install pydantic httpx opencv-python
```

(asyncio is stdlib)
