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
        if not any(c.rank == Rank.ACE for c in self.cards):
            return False
        # Check if we have an ace that's still counting as 11
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == Rank.ACE)
        reduced = 0
        while total > 21 and aces:
            total -= 10
            aces -= 1
            reduced += 1
        # Soft if we have aces and didn't reduce all of them
        total_aces = sum(1 for c in self.cards if c.rank == Rank.ACE)
        return reduced < total_aces and total <= 21

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
    WAITING_FOR_DEAL = "waiting_for_deal"
    PLAYER_TURN = "player_turn"
    DEALER_TURN = "dealer_turn"
    HAND_COMPLETE = "hand_complete"


class Action(str, Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"


class GameState(BaseModel):
    dealer_hand: Hand = Hand()
    player_hand: Hand = Hand()
    dealer_hole_card_hidden: bool = True
    phase: GamePhase = GamePhase.WAITING_FOR_DEAL
    last_action: Action | None = None


class ActionOutput(BaseModel):
    """JSON output for VLA robot"""
    action: Action
    confidence: float
    commentary: str
