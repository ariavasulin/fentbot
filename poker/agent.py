from openai import OpenAI
from pydantic import BaseModel

from poker.config import OPENROUTER_API_KEY
from poker.models import GameState, Action
from poker.prompts import PETER_SYSTEM_PROMPT, format_game_state
from poker.strategy import basic_strategy, get_dealer_upcard_value


def get_client() -> OpenAI:
    """Get OpenAI client (lazy initialization)"""
    return OpenAI(
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
    client = get_client()
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
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
    client = get_client()
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {"role": "system", "content": PETER_SYSTEM_PROMPT},
            {"role": "user", "content": f"React to this: {event}"}
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
