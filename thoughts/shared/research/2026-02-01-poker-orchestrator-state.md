---
date: 2026-02-01T12:00:00-08:00
researcher: claude
git_commit: c9ca43d8ef93d0160449b1daf7d9927cdef4877c
branch: main
repository: fentbot
topic: "Poker Orchestrator Current State and Action Triggering Mechanism"
tags: [research, codebase, poker, vision, orchestrator, opencv]
status: complete
last_updated: 2026-02-01
last_updated_by: claude
---

# Research: Poker Orchestrator Current State and Action Triggering Mechanism

**Date**: 2026-02-01T12:00:00-08:00
**Researcher**: claude
**Git Commit**: c9ca43d8ef93d0160449b1daf7d9927cdef4877c
**Branch**: main
**Repository**: fentbot

## Research Question
What is the current state of the poker orchestrator? How are actions triggered? The user wants to use OpenCV to auto-detect significant movement instead of acting every 5 seconds regardless of board state.

## Summary

The poker orchestrator is a **blackjack automation system** (despite the "poker" naming) located in `/Users/ariasulin/Git/fentbot/poker/`. It uses a continuous polling loop that captures camera frames every 0.5 seconds (not 5 seconds as mentioned) and sends them to Gemini 2.0 Flash Lite for card recognition. Actions are triggered when the AI-detected card state changes, not on a fixed timer.

**Current approach**: The system polls continuously but only takes action when `cards_changed()` returns `True` - meaning the detected cards differ from the previous frame. This is **semantic change detection** (comparing parsed card objects) rather than **visual change detection** (comparing raw pixels/images).

## Detailed Findings

### Main Orchestrator (`poker/main.py`)

The `run()` function implements the main game loop:

```python
while True:
    frame = capture_frame(CAMERA_INDEX)      # Always captures
    reading = read_cards(frame)               # Always calls Gemini API

    if not cards_changed(last_reading, reading):  # Semantic comparison
        time.sleep(FRAME_INTERVAL)            # 0.5 second delay
        continue                               # Skip action

    # ... process phase and take action ...
```

**Key observation**: Every iteration makes an API call to Gemini regardless of whether the visual scene changed. The `cards_changed()` function compares the *parsed results* (Card objects), not the raw images.

### Change Detection (`poker/main.py:12-19`)

```python
def cards_changed(old_reading: TableReading | None, new_reading: TableReading) -> bool:
    if old_reading is None:
        return True
    return (
        old_reading.dealer_cards != new_reading.dealer_cards or
        old_reading.player_cards != new_reading.player_cards
    )
```

This compares lists of `Card` objects (rank/suit). It does NOT compare:
- Raw image pixels
- Visual changes in the frame
- Motion/movement in the scene

### Vision System (`poker/vision.py`)

- `capture_frame(camera_index)`: Opens camera, captures single frame, immediately releases
- `frame_to_base64(frame)`: Encodes frame as JPEG base64 for API
- `read_cards(frame)`: Sends frame to Gemini 2.0 Flash Lite via OpenRouter API

**Current limitation**: Every frame capture triggers an API call. There is no local preprocessing to detect if the frame is worth analyzing.

### Configuration (`poker/config.py`)

```python
CAMERA_INDEX = 0
FRAME_INTERVAL = 0.5  # seconds between captures
```

The 0.5 second interval is the *minimum* delay between iterations when no cards change. When cards do change, processing happens immediately.

### Action Triggering Flow

1. **Continuous polling**: `while True` loop runs indefinitely
2. **Frame capture**: Every iteration captures a frame (no motion gating)
3. **API call**: Every frame is sent to Gemini for card recognition
4. **Semantic comparison**: Parsed card objects are compared to previous reading
5. **Conditional action**: If cards changed, determine game phase and respond

### Game Phases and Responses

| Phase | Condition | Action |
|-------|-----------|--------|
| `WAITING_FOR_DEAL` | No cards visible | Speak "new hand" if cards were cleared |
| `PLAYER_TURN` | Cards visible, dealer hole hidden | Get Peter's decision via LLM, speak commentary |
| `DEALER_TURN` | Dealer hole revealed | Evaluate hand, get reaction via LLM |

### What Does NOT Exist

- **Motion detection**: No OpenCV-based frame differencing
- **Image comparison**: No pixel-level change detection
- **Background subtraction**: No cv2.createBackgroundSubtractorMOG2()
- **Movement thresholds**: No configurable sensitivity for visual changes
- **Frame buffering**: No comparison of current vs previous frames at image level

## Code References

- `poker/main.py:34-126` - Main `run()` function with game loop
- `poker/main.py:12-19` - `cards_changed()` semantic comparison function
- `poker/main.py:45-54` - Core polling loop structure
- `poker/vision.py:28-35` - `capture_frame()` function
- `poker/vision.py:44-106` - `read_cards()` API call to Gemini
- `poker/config.py:11-12` - Camera and interval configuration
- `poker/models.py:88-92` - `GamePhase` enum definitions
- `poker/agent.py:23-62` - `get_peter_decision()` for LLM commentary

## Architecture Documentation

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Loop (main.py)                     │
│                                                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │ Camera  │───▶│ Gemini   │───▶│ Semantic Comparison  │   │
│  │ Capture │    │ API Call │    │ (cards_changed)      │   │
│  └─────────┘    └──────────┘    └──────────────────────┘   │
│       │                                    │                │
│       │ Every 0.5s                         │ If changed     │
│       ▼                                    ▼                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Phase Detection & Action               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ Waiting    │  │ Player     │  │ Dealer     │    │   │
│  │  │ for Deal   │  │ Turn       │  │ Turn       │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Output (voice.py, stdout)              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Inefficiency in Current Design

Every loop iteration:
1. Opens camera connection
2. Captures frame
3. Closes camera connection
4. Encodes frame as JPEG/base64
5. Makes HTTP request to OpenRouter API
6. Parses JSON response
7. Compares parsed objects

This happens regardless of whether anything visually changed in the scene.

## Historical Context (from thoughts/)

No prior research documents exist specifically about the poker orchestrator's action triggering mechanism.

Related planning documents:
- `thoughts/shared/plans/2025-01-31-blackjack-vision-peter-griffin.md` - Original implementation plan
- `thoughts/shared/plans/poker-vision-tracker.md` - Vision tracking planning

## Implications for OpenCV Motion Detection

To implement motion-based triggering, you would need to:

1. **Keep camera open**: Instead of open/close per frame, maintain persistent `VideoCapture`
2. **Buffer previous frame**: Store the last captured frame in memory
3. **Frame differencing**: Use OpenCV to compare current vs previous frame
4. **Threshold check**: Only proceed to API call if motion exceeds threshold

Relevant OpenCV functions that don't currently exist in the codebase:
- `cv2.absdiff()` - Compute absolute difference between frames
- `cv2.threshold()` - Binary threshold for motion detection
- `cv2.countNonZero()` - Count changed pixels
- `cv2.createBackgroundSubtractorMOG2()` - Background subtraction

## Open Questions

1. What constitutes "significant movement" in this blackjack context? (Card dealing, chip movement, hand gestures?)
2. Should motion detection be a pre-filter before API calls, or replace semantic comparison entirely?
3. What motion threshold would be appropriate to detect card deals but ignore minor vibrations/noise?
