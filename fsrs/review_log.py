
"""
review_log
---------

This module defines the ReviewLog and Rating classes.

Classes:
    ReviewLog: Represents the log entry of a Topic that has been reviewed.
    Rating: Enum representing the four possible ratings when reviewing a topic.
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime


class Rating(IntEnum):
    """
    Enum representing the four possible ratings when reviewing a topic.
    """

    Again = 1
    Hard = 2
    Good = 3
    Easy = 4


@dataclass
class ReviewLog:
    """
    Represents the log entry of a Topic object that has been reviewed.

    Attributes:
        topic_id: The id of the topic being reviewed.
        rating: The rating given to the topic during the review.
        review_datetime: The date and time of the review.
        review_duration: The number of miliseconds it took to review the topic or None if unspecified.
    """

    topic_id: int
    rating: Rating
    review_datetime: datetime
    review_duration: int | None

    def to_dict(
        self,
    ) -> dict[str, dict | int | str | None]:
        """
        Returns a JSON-serializable dictionary representation of the ReviewLog object.

        This method is specifically useful for storing ReviewLog objects in a database.

        Returns:
            A dictionary representation of the ReviewLog object.
        """

        return_dict = {
            "topic_id": self.topic_id,
            "rating": self.rating.value,
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
        }

        return return_dict

    @staticmethod
    def from_dict(
        source_dict: dict[str, dict | int | str | None],
    ) -> ReviewLog:
        """
        Creates a ReviewLog object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing ReviewLog object.

        Returns:
            A ReviewLog object created from the provided dictionary.
        """

        topic_id = source_dict["topic_id"]
        rating = Rating(int(source_dict["rating"]))
        review_datetime = datetime.fromisoformat(source_dict["review_datetime"])
        review_duration = source_dict["review_duration"]

        return ReviewLog(
            topic_id=topic_id,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )


__all__ = ["ReviewLog", "Rating"]
