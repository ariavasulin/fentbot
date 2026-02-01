# Blackjack Demo: Betting + Robot Integration

## Overview

Add betting phase and robot integration to the existing blackjack vision system. Peter bets a $50 yellow chip at the start of each hand, robot places it, system verifies placement and reacts if it fails.

## Current State Analysis

**What already works** (`poker/main.py`):
- Polls camera every 0.5s
- `cards_changed()` prevents redundant actions
- Phase detection (WAITING_FOR_DEAL, PLAYER_TURN, DEALER_TURN)
- HIT/STAND decisions with Peter commentary
- Win/lose reactions

**What's missing**:
- Betting phase (Peter doesn't bet, just watches)
- Robot integration (no way to signal robot or know when it's done)
- Chip verification (no check if robot succeeded)

### Key Files:
- `poker/main.py:34-129` - Main loop
- `poker/vision.py:44-106` - Gemini card reading
- `poker/config.py` - Settings

## Desired End State

1. When WAITING_FOR_DEAL and no bet placed → Peter says "let's bet $50", outputs robot command
2. System keeps polling during robot action (can detect failures mid-action)
3. If chip drops/fails mid-action → Peter says zinger immediately
4. After robot done → verify chip placement via Gemini
5. If chip succeeded → wait for cards to be dealt
6. Rest of hand plays as normal (HIT/STAND verbal, win/lose reactions)

### Verification:
- Peter bets at start of each hand
- Robot command JSON appears on stdout
- After robot done, chip verification runs
- Failed placement triggers zinger
- Hand plays through normally after successful bet

## What We're NOT Doing

- Motion detection (polling is fine)
- Multiple chip denominations (just $50 yellow)
- Robot actions for HIT/DOUBLE/SPLIT (verbal only)
- Keeping camera open continuously (current open/close is fine)

## Implementation Approach

Minimal changes:
1. Add `verify_chip_placement()` to `vision.py`
2. Add `robot_executing` flag and betting logic to `main.py`
3. Create simple `robot_interface.py` for external robot to signal done

---

## Phase 1: Chip Verification

### Overview
Add Gemini-based chip placement verification to vision module.

### Changes Required:

#### 1. Add to `poker/vision.py`

After the existing `TableReading` class (around line 26), add:

```python
class ChipVerification(BaseModel):
    """Result of chip placement verification."""
    chip_in_betting_area: bool
    chip_visible: bool
    failure_description: str | None = None
```

After the existing `read_cards()` function (after line 106), add:

```python
def verify_chip_placement(frame: np.ndarray) -> ChipVerification:
    """
    Check if the yellow $50 chip made it to the betting area.
    Called immediately after robot finishes betting action.
    """
    b64_image = frame_to_base64(frame)

    client = get_client()
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "system",
                "content": """You are verifying a robot's chip placement in blackjack.
Look for a YELLOW $50 chip. Check if it's in the betting circle/area.

Respond in this exact JSON format:
{
    "chip_in_betting_area": true/false,
    "chip_visible": true/false,
    "failure_description": "description if failed, null if success"
}

Examples of failures:
- Chip fell off table
- Chip is outside the betting area
- Chip knocked over other chips
- Cannot see the yellow chip anywhere"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    },
                    {
                        "type": "text",
                        "text": "Did the yellow $50 chip make it to the betting area?"
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
    )

    data = json.loads(response.choices[0].message.content)
    return ChipVerification(**data)
```

### Success Criteria:

#### Automated Verification:
- [ ] Import works: `python -c "from poker.vision import verify_chip_placement, ChipVerification"`

#### Manual Verification:
- [ ] Test with camera showing chip in betting area → returns `chip_in_betting_area=True`

---

## Phase 2: Robot Interface

### Overview
Simple module for robot to signal when it's done. Robot integration sets a flag before acting, clears when done.

### Changes Required:

#### 1. New file: `poker/robot_interface.py`

```python
"""
Simple robot integration interface.

Usage from robot controller:
    from poker.robot_interface import robot_started, robot_done

    robot_started()
    # ... do robot action ...
    robot_done()
"""

import threading

_robot_executing = False
_robot_done_event = threading.Event()
_lock = threading.Lock()


def is_robot_executing() -> bool:
    """Check if robot is currently executing an action."""
    with _lock:
        return _robot_executing


def robot_started() -> None:
    """Call when robot begins an action."""
    global _robot_executing
    with _lock:
        _robot_executing = True
        _robot_done_event.clear()
    print("[ROBOT] Action started")


def robot_done() -> None:
    """Call when robot finishes an action."""
    global _robot_executing
    with _lock:
        _robot_executing = False
        _robot_done_event.set()
    print("[ROBOT] Action complete")


```

### Success Criteria:

#### Automated Verification:
- [ ] Import works: `python -c "from poker.robot_interface import robot_started, robot_done, is_robot_executing"`

#### Manual Verification:
- [ ] `is_robot_executing()` returns False initially
- [ ] After `robot_started()`, returns True
- [ ] After `robot_done()`, returns False

---

## Phase 3: Main Loop Updates

### Overview
Add betting phase and robot integration to main loop.

### Changes Required:

#### 1. Update imports in `poker/main.py`

Change line 8 from:
```python
from poker.vision import TableReading, capture_frame, read_cards
```
to:
```python
from poker.vision import TableReading, capture_frame, read_cards, verify_chip_placement
```

Add import:
```python
from poker.robot_interface import is_robot_executing, robot_started, robot_done
```

#### 2. Add `bet_placed` flag

After line 43 (`last_decision_made = False`), add:
```python
    bet_placed = False
```

#### 3. Add betting logic in WAITING_FOR_DEAL phase

Replace lines 67-70:
```python
            if game_state.phase == GamePhase.WAITING_FOR_DEAL:
                if last_reading is not None:  # Cards were cleared
                    speak_async("Alright, new hand! Let's go!")
                last_decision_made = False
```

With:
```python
            if game_state.phase == GamePhase.WAITING_FOR_DEAL:
                if not bet_placed and not is_robot_executing():
                    # Time to bet!
                    print("\n>>> BETTING: Placing $50 chip")
                    speak("Alright, let's put fifty bucks on this one!")

                    # Signal robot to bet
                    robot_started()
                    bet_output = {
                        "action": "BET",
                        "amount": 50,
                        "chip_color": "yellow"
                    }
                    print(f"ROBOT_ACTION: {json.dumps(bet_output)}")
                    # Robot will call robot_done() when finished
                    # Polling continues - we'll verify on next iteration after robot_done

                elif is_robot_executing():
                    # Robot is moving - check for failures mid-action
                    frame = capture_frame(CAMERA_INDEX)
                    verification = verify_chip_placement(frame)

                    if not verification.chip_visible and not verification.chip_in_betting_area:
                        # Chip dropped mid-action!
                        print("  Chip DROPPED mid-action!")
                        zinger = get_reaction(f"The robot dropped the chip! {verification.failure_description}")
                        speak(zinger)

                elif not bet_placed:
                    # Robot just finished (not executing, bet not placed) - verify final placement
                    frame = capture_frame(CAMERA_INDEX)
                    verification = verify_chip_placement(frame)

                    if verification.chip_in_betting_area:
                        print("  Chip placement: SUCCESS")
                        bet_placed = True
                    else:
                        print(f"  Chip placement: FAILED - {verification.failure_description}")
                        zinger = get_reaction(f"The robot screwed up! {verification.failure_description}")
                        speak(zinger)
                        # Don't set bet_placed - will retry next loop

                elif last_reading is not None and cards_changed(last_reading, reading):
                    # Cards were cleared, new hand starting
                    speak_async("Alright, new hand! Let's go!")
                    bet_placed = False

                last_decision_made = False
```

#### 5. Reset bet_placed after hand completes

After line 113 (`last_decision_made = False  # Reset for next hand`), add:
```python
                bet_placed = False
```

### Success Criteria:

#### Automated Verification:
- [ ] No syntax errors: `python -m py_compile poker/main.py`
- [ ] Import works: `python -c "from poker.main import run"`

#### Manual Verification:
- [ ] Run system → Peter bets when WAITING_FOR_DEAL
- [ ] ROBOT_ACTION JSON printed to stdout
- [ ] System keeps polling during robot action
- [ ] If chip drops mid-action, Peter says zinger immediately
- [ ] After `robot_done()` called, chip verification runs
- [ ] If chip failed, Peter says zinger
- [ ] Hand plays normally after successful bet
- [ ] bet_placed resets after hand completes

---

## Testing Strategy

### Manual Testing Steps:
1. Start system with camera on empty table
2. Verify Peter attempts to bet (ROBOT_ACTION printed)
3. While robot executing, remove chip from view → verify Peter says zinger
4. In another terminal: `python -c "from poker.robot_interface import robot_done; robot_done()"`
5. Verify final chip verification runs
6. Place cards in view → verify HIT/STAND works
7. Complete hand → verify bet_placed resets

### Integration Test:
1. Full hand cycle: bet → deal → decide → outcome → new bet

## References

- Current main loop: `poker/main.py:34-129`
- Card reading: `poker/vision.py:44-106`
- Phase detection: `poker/main.py:22-31`
- Existing decision blocking: `poker/main.py:73`
