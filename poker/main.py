import json
import time
from datetime import datetime

from poker.agent import generate_command, get_peter_decision, get_reaction
from poker.config import CAMERA_INDEX, FRAME_INTERVAL
from poker.models import ActionOutput, GamePhase, GameState, Hand
<<<<<<< HEAD
from poker.vision import TableReading, capture_frame, read_cards
from poker.robot import Robot
=======
from poker.vision import TableReading, capture_frame, read_cards, verify_chip_placement
from poker.robot_interface import is_robot_executing, robot_started, robot_done
>>>>>>> d05b97f (Zingers)
from poker.voice import speak, speak_async


def cards_changed(old_reading: TableReading | None, new_reading: TableReading) -> bool:
    """Check if cards on table have changed"""
    if old_reading is None:
        return True
    return (
        old_reading.dealer_cards != new_reading.dealer_cards or
        old_reading.player_cards != new_reading.player_cards
    )


def determine_phase(reading: TableReading) -> GamePhase:
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

    last_reading: TableReading | None = None
    game_state = GameState()
    last_decision_made = False
<<<<<<< HEAD
    robot = Robot()
=======
    bet_placed = False
>>>>>>> d05b97f (Zingers)

    while True:
        try:
            # Wait for robot command to complete
            if robot.command_in_progress():
                time.sleep(FRAME_INTERVAL)
                continue

            # Capture and analyze
            frame = capture_frame(CAMERA_INDEX)
            reading = read_cards(frame)

            # Check if anything changed (but keep polling during betting phase)
            needs_betting_check = is_robot_executing() or (not bet_placed and determine_phase(reading) == GamePhase.WAITING_FOR_DEAL)
            if not cards_changed(last_reading, reading) and not needs_betting_check:
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

                        # Generate and execute a chip placement command (single chip)
                        command = generate_command(reading, response.action)
                        if command and not robot.command_in_progress():
                            print(f"ROBOT COMMAND: {command}")
                            robot.execute_command(command)

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
                bet_placed = False

            last_reading = reading

        except KeyboardInterrupt:
            print("\nShutting down...")
            speak("Alright, I'm out. Lois is gonna kill me when she sees the credit card bill.")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    run()
