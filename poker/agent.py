from openai import OpenAI
from pydantic import BaseModel

from poker.config import OPENROUTER_API_KEY
from typing import TYPE_CHECKING

from poker.models import Action, Chip, ChipColor, GameState
from poker.prompts import PETER_SYSTEM_PROMPT, format_game_state
from poker.strategy import basic_strategy, get_dealer_upcard_value, simplify_action

if TYPE_CHECKING:
    from poker.vision import TableReading


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
    # Get the correct action from basic strategy (simplified to HIT/STAND only)
    dealer_upcard = get_dealer_upcard_value(game_state.dealer_hand)
    correct_action = simplify_action(basic_strategy(game_state.player_hand, dealer_upcard))

    # Format cards for display
    dealer_cards_str = ", ".join(str(c) for c in game_state.dealer_hand.cards)
    if game_state.dealer_hole_card_hidden:
        dealer_cards_str += " + [HIDDEN]"
    player_cards_str = ", ".join(str(c) for c in game_state.player_hand.cards)

    # Get Peter's commentary
    # client = get_client()
    # response = client.chat.completions.create(
    #     model="google/gemini-2.0-flash-lite-001",
    #     messages=[
    #         {"role": "system", "content": PETER_SYSTEM_PROMPT},
    #         {
    #             "role": "user",
    #             "content": format_game_state(
    #                 dealer_cards_str,
    #                 player_cards_str,
    #                 game_state.player_hand.value,
    #                 dealer_upcard
    #             ) + f"\n\n[The correct play is: {correct_action.value.upper()}]"
    #         }
    #     ],
    #     max_tokens=150,
    # )

    # commentary = response.choices[0].message.content.strip()

    return PeterResponse(
        action=correct_action,
        commentary="Peter's commentary"
    )


def get_reaction(event: str) -> str:
    """Get Peter's reaction to game events (dealer bust, win, lose, etc.)"""
    # client = get_client()
    # response = client.chat.completions.create(
    #     model="google/gemini-2.0-flash-lite-001",
    #     messages=[
    #         {"role": "system", "content": PETER_SYSTEM_PROMPT},
    #         {"role": "user", "content": f"React to this: {event}"}
    #     ],
    #     max_tokens=100,
    # )
    # return response.choices[0].message.content.strip()
    return "Peter's reaction"


CHIP_VALUE_MAP: dict[ChipColor, int] = {
    ChipColor.PINK: 1,
    ChipColor.GREEN: 5,
    ChipColor.RED: 10,
    ChipColor.YELLOW: 50,
}


def _select_closest_chip(chips: list[Chip], target_amount: int) -> Chip | None:
    """
    Pick one available chip whose value is closest to the target.
    If tied, choose the lower-value chip to avoid over-betting.
    """
    if not chips:
        return None

    def chip_sort_key(chip: Chip) -> tuple[int, int]:
        value = CHIP_VALUE_MAP.get(chip.color, 0)
        return (abs(value - target_amount), value)

    return sorted(chips, key=chip_sort_key)[0]


def _recommended_bet_for_action(action: Action) -> int:
    """
    Heuristic bet sizing tied to the strategy outcome.
    - Double / Split: recommend 10
    - Otherwise: recommend 5
    """
    if action in (Action.DOUBLE, Action.SPLIT):
        return 10
    return 5


def generate_command(reading: "TableReading", action: Action) -> str | None:
    """
    Create a natural-language command for the VLA to place a single chip.

    Picks one chip (max 1) with value closest to the recommended bet derived
    from the strategy outcome.
    """
    target_amount = _recommended_bet_for_action(action)
    chip = _select_closest_chip(reading.chips, target_amount)
    if chip is None:
        return None

    return (
        f"pickup the {chip.position.value} {chip.color.value} chip and place it in the black square"
    )
