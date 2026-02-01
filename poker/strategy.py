from poker.models import Hand, Action, Rank


def get_dealer_upcard_value(dealer_hand: Hand) -> int:
    """Get value of dealer's visible card"""
    if not dealer_hand.cards:
        return 0
    return dealer_hand.cards[0].value


def basic_strategy(player_hand: Hand, dealer_upcard: int) -> Action:
    """
    Basic strategy lookup.
    Returns optimal action based on player hand and dealer upcard.
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
