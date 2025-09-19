from enum import IntEnum
import math

# USER_NAME = 'Mrs.Krishna'
USER_NAME = 'User'

class State(IntEnum):
    """Learning state of a Topic object"""
    Learning = 1
    Review = 2
    Relearning = 3

class Rating(IntEnum):
    """Four possible ratings when reviewing a topic"""
    Again = 1
    Hard = 2
    Good = 3
    Easy = 4

# FSRS Constants
FSRS_DEFAULT_DECAY = 0.1542
DEFAULT_PARAMETERS = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
    1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
    1.8729, 0.5425, 0.0912, 0.0658, FSRS_DEFAULT_DECAY,
)

STABILITY_MIN = 0.001
MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
INITIAL_STABILITY_MAX = 100.0

FUZZ_RANGES = [
    {"start": 2.5, "end": 7.0, "factor": 0.15},
    {"start": 7.0, "end": 20.0, "factor": 0.1},
    {"start": 20.0, "end": math.inf, "factor": 0.05},
]

