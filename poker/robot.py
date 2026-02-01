"""
Original robot implementation commented out for testing.
"""

import threading
from typing import Optional


class Robot:
    """
    Fake robot used for local testing.
    execute_command marks the robot busy for ~5 seconds, then clears.
    """

    def __init__(self) -> None:
        self._in_progress: bool = False
        self._timer: Optional[threading.Timer] = None

    def _finish(self) -> None:
        self._in_progress = False
        self._timer = None

    def execute_command(self, command: str) -> None:
        """
        Simulate sending a natural-language command to the robot arm.

        Example: "pickup the left green chip and place it in the black square"
        """
        self._in_progress = True
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(5.0, self._finish)
        self._timer.daemon = True
        self._timer.start()

    def command_in_progress(self) -> bool:
        """Return True if the robot is currently executing a command."""
        return self._in_progress

