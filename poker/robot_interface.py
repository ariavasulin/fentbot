"""
Simple robot integration interface.

Usage from robot controller:
    from poker.robot_interface import robot_started, robot_done

    robot_started()
    # ... do robot action ...
    robot_done()

State is stored in a file so it can be shared across processes (for testing).
"""

import os

_STATE_FILE = "/tmp/robot_executing.flag"


def is_robot_executing() -> bool:
    """Check if robot is currently executing an action."""
    return os.path.exists(_STATE_FILE)


def robot_started() -> None:
    """Call when robot begins an action."""
    with open(_STATE_FILE, "w") as f:
        f.write("1")
    print("[ROBOT] Action started")


def robot_done() -> None:
    """Call when robot finishes an action."""
    if os.path.exists(_STATE_FILE):
        os.remove(_STATE_FILE)
    print("[ROBOT] Action complete")
