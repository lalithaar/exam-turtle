"""
topic
---------

This module defines the Topic and State classes.

Classes:
    Topic: Represents a flashtopic in the FSRS system.
    State: Enum representing the learning state of a Topic object.
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime, timezone
import time


class State(IntEnum):
    """
    Enum representing the learning state of a Topic object.
    """

    Learning = 1
    Review = 2
    Relearning = 3


@dataclass(init=False)
class Topic:
    """
    Represents a flashtopic in the FSRS system.

    Attributes:
        topic_id: The id of the topic. Defaults to the epoch milliseconds of when the topic was created.
        state: The topic's current learning state.
        step: The topic's current learning or relearning step or None if the topic is in the Review state.
        stability: Core mathematical parameter used for future scheduling.
        difficulty: Core mathematical parameter used for future scheduling.
        due: The date and time when the topic is due next.
        last_review: The date and time of the topic's last review.
    """

    topic_id: int
    state: State
    step: int | None
    stability: float | None
    difficulty: float | None
    due: datetime
    last_review: datetime | None

    def __init__(
        self,
        topic_id: int | None = None,
        state: State = State.Learning,
        step: int | None = None,
        stability: float | None = None,
        difficulty: float | None = None,
        due: datetime | None = None,
        last_review: datetime | None = None,
    ) -> None:
        if topic_id is None:
            # epoch milliseconds of when the topic was created
            topic_id = int(datetime.now(timezone.utc).timestamp() * 1000)
            # wait 1ms to prevent potential topic_id collision on next Topic creation
            time.sleep(0.001)
        self.topic_id = topic_id

        self.state = state

        if self.state == State.Learning and step is None:
            step = 0
        self.step = step

        self.stability = stability
        self.difficulty = difficulty

        if due is None:
            due = datetime.now(timezone.utc)
        self.due = due

        self.last_review = last_review

    def to_dict(self) -> dict[str, int | float | str | None]:
        """
        Returns a JSON-serializable dictionary representation of the Topic object.

        This method is specifically useful for storing Topic objects in a database.

        Returns:
            A dictionary representation of the Topic object.
        """

        return_dict = {
            "topic_id": self.topic_id,
            "state": self.state.value,
            "step": self.step,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "due": self.due.isoformat(),
            "last_review": self.last_review.isoformat() if self.last_review else None,
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, int | float | str | None]) -> Topic:
        """
        Creates a Topic object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing Topic object.

        Returns:
            A Topic object created from the provided dictionary.
        """

        topic_id = int(source_dict["topic_id"])
        state = State(int(source_dict["state"]))
        step = source_dict["step"]
        stability = (
            float(source_dict["stability"]) if source_dict["stability"] else None
        )
        difficulty = (
            float(source_dict["difficulty"]) if source_dict["difficulty"] else None
        )
        due = datetime.fromisoformat(source_dict["due"])
        last_review = (
            datetime.fromisoformat(source_dict["last_review"])
            if source_dict["last_review"]
            else None
        )

        return Topic(
            topic_id=topic_id,
            state=state,
            step=step,
            stability=stability,
            difficulty=difficulty,
            due=due,
            last_review=last_review,
        )


__all__ = ["Topic", "State"]