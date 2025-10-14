"""
Optimized Database Models for FSRS Algorithm Integration
======================================================

This models.py is designed to efficiently serve both:
1. Backend algorithm calculations (comprehensive metadata)
2. User-facing features (clean, meaningful data presentation)

Key Design Principles:
- Single user system (no user authentication needed)
- Efficient data storage with proper indexing
- Clear separation between internal algorithm data and user-visible metrics
- Optimized for the robust algorithm.py calculations
- Student-friendly data presentation methods
"""

from db import db
from datetime import datetime, timezone, date, timedelta
from typing import List, Dict, Optional, Tuple
import statistics
import json
from enum import IntEnum
from dataclasses import asdict

# Import our algorithm components
from services.algorithm import (
    TopicMemory,
    TopicState,
    Rating,
    CognitiveState,
    ComprehensiveTopicScheduler,
    StudySession,
)
from utils.datetime_utils import ensure_timezone_aware, now_ist


class ReviewRating(IntEnum):
    """User-friendly rating scale (maps to algorithm.Rating)"""

    BLACKOUT = 1  # "I had no idea"
    AGAIN = 2  # "I recognized it but couldn't recall"
    HARD = 3  # "I got it but it was difficult"
    GOOD = 4  # "I recalled it normally"
    EASY = 5  # "I knew it immediately"


class Topic(db.Model):
    """
    Core topic model with both algorithm metadata and user-facing features
    """

    __tablename__ = "topics"

    # Primary identification
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, index=True)
    subject = db.Column(db.String(100), nullable=True, index=True)
    description = db.Column(db.Text, nullable=True)

    # Algorithm-specific metadata (hidden from user)
    fsrs_stability = db.Column(db.Float, default=1.0, nullable=False)
    fsrs_difficulty = db.Column(db.Float, default=5.0, nullable=False)
    algorithm_state = db.Column(db.Integer, default=TopicState.NEW, nullable=False)

    # User-meaningful metrics
    mastery_level = db.Column(
        db.String(20), default="beginner", nullable=False
    )  # beginner/developing/proficient/advanced
    confidence_score = db.Column(db.Float, default=0.0, nullable=False)  # 0.0 to 1.0
    last_performance = db.Column(
        db.String(20), nullable=True
    )  # excellent/good/fair/poor/critical

    # Learning progress tracking
    total_study_time_minutes = db.Column(db.Integer, default=0, nullable=False)
    total_reviews = db.Column(db.Integer, default=0, nullable=False)
    successful_reviews = db.Column(db.Integer, default=0, nullable=False)
    current_streak = db.Column(db.Integer, default=0, nullable=False)
    best_streak = db.Column(db.Integer, default=0, nullable=False)
    times_forgotten = db.Column(db.Integer, default=0, nullable=False)

    # Scheduling and timing
    next_review_date = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    last_reviewed_date = db.Column(db.DateTime(timezone=True), nullable=True)
    first_learned_date = db.Column(db.DateTime(timezone=True), nullable=True)

    # Content and complexity
    complexity_rating = db.Column(db.Float, default=5.0, nullable=False)  # 1-10 scale
    tags = db.Column(db.Text, nullable=True)  # JSON array of tags
    notes = db.Column(db.Text, nullable=True)  # User notes

    # Performance analytics (derived fields updated by triggers/methods)
    average_response_time = db.Column(db.Float, nullable=True)  # seconds
    recent_performance_trend = db.Column(
        db.String(20), default="stable", nullable=False
    )  # improving/declining/stable
    estimated_next_success_rate = db.Column(
        db.Float, nullable=True
    )  # Algorithm prediction

    # Timestamps
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    review_sessions = db.relationship(
        "ReviewSession", backref="topic", lazy="dynamic", cascade="all, delete-orphan"
    )
    exam_associations = db.relationship(
        "ExamTopic", backref="topic", lazy="dynamic", cascade="all, delete-orphan"
    )

    def __init__(self, name: str, subject: str = None, **kwargs):
        super().__init__()
        self.name = name
        self.subject = subject
        self.first_learned_date = now_ist()
        self.next_review_date = now_ist()

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # =========================================================================
    # ALGORITHM INTEGRATION METHODS
    # =========================================================================

    def to_algorithm_memory(self) -> TopicMemory:
        """Convert SQLAlchemy model to algorithm TopicMemory object"""
        # Get recent review ratings
        recent_sessions = (
            self.review_sessions.order_by(ReviewSession.reviewed_at.desc())
            .limit(10)
            .all()
        )

        recent_ratings = [Rating(session.rating) for session in recent_sessions]
        response_times = [
            session.response_time_seconds or 5.0
            for session in recent_sessions
            if session.response_time_seconds
        ]

        # Get exam dates
        exam_dates = [
            assoc.exam.exam_date
            for assoc in self.exam_associations
            if assoc.exam.exam_date >= date.today()
        ]
        exam_weights = [
            assoc.importance_weight
            for assoc in self.exam_associations
            if assoc.exam.exam_date >= date.today()
        ]

        # Count cramming sessions (reviews within 2 days of each other)
        cramming_count = 0
        sorted_sessions = sorted(recent_sessions, key=lambda x: x.reviewed_at)
        for i in range(1, len(sorted_sessions)):
            time_diff = (
                sorted_sessions[i].reviewed_at - sorted_sessions[i - 1].reviewed_at
            ).total_seconds()
            if time_diff < 172800:  # 2 days in seconds
                cramming_count += 1

        return TopicMemory(
            stability=self.fsrs_stability,
            difficulty=self.fsrs_difficulty,
            initial_learning_date=self.first_learned_date,
            last_review_date=self.last_reviewed_date,
            due_date=self.next_review_date,
            state=TopicState(self.algorithm_state),
            review_count=self.total_reviews,
            lapses=self.times_forgotten,
            streak=self.current_streak,
            average_retention=self.successful_reviews / max(1, self.total_reviews),
            response_times=response_times,
            recent_ratings=recent_ratings,
            similar_topics=[],  # Could be enhanced with topic similarity analysis
            subject_category=self.subject,
            complexity_score=self.complexity_rating,
            exam_dates=exam_dates,
            exam_weights=exam_weights,
            cramming_sessions=cramming_count,
        )

    def update_from_algorithm_result(self, algorithm_result: Dict):
        """Update model from algorithm processing result"""
        memory = algorithm_result["updated_memory"]

        # Update algorithm metadata
        self.fsrs_stability = memory.stability
        self.fsrs_difficulty = memory.difficulty
        self.algorithm_state = memory.state.value
        self.next_review_date = memory.due_date
        self.last_reviewed_date = memory.last_review_date
        self.total_reviews = memory.review_count
        self.times_forgotten = memory.lapses
        self.current_streak = memory.streak

        # Update user-facing metrics
        self.successful_reviews = sum(
            1 for r in memory.recent_ratings if r >= Rating.GOOD
        )
        self.best_streak = max(self.best_streak, memory.streak)

        # Update performance indicators
        strength = algorithm_result.get("strength_analysis", {})
        self.confidence_score = strength.get(
            "prediction_confidence", self.confidence_score
        )
        self.last_performance = strength.get(
            "readiness_category", self.last_performance
        )
        self.estimated_next_success_rate = strength.get(
            "exam_adjusted_retrievability", None
        )

        # Update mastery level based on algorithm state and performance
        self.mastery_level = self._calculate_mastery_level(memory, strength)
        self.recent_performance_trend = self._calculate_performance_trend()

        self.updated_at = now_ist()

    def _calculate_mastery_level(self, memory: TopicMemory, strength: Dict) -> str:
        """Calculate user-friendly mastery level"""
        maturity = strength.get("maturity_score", 0.0)
        retrievability = strength.get("exam_adjusted_retrievability", 0.0)

        if (
            memory.state == TopicState.MATURE
            and maturity >= 0.8
            and retrievability >= 0.85
        ):
            return "advanced"
        elif memory.state in [TopicState.REVIEW, TopicState.MATURE] and maturity >= 0.6:
            return "proficient"
        elif (
            memory.state in [TopicState.YOUNG, TopicState.REVIEW] and memory.streak >= 3
        ):
            return "developing"
        else:
            return "beginner"

    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        recent_sessions = (
            self.review_sessions.order_by(ReviewSession.reviewed_at.desc())
            .limit(6)
            .all()
        )

        if len(recent_sessions) < 4:
            return "stable"

        # Split into recent vs older halves
        recent_half = recent_sessions[:3]
        older_half = recent_sessions[3:6]

        recent_avg = statistics.mean(
            [
                s.retention_percentage
                for s in recent_half
                if s.retention_percentage is not None
            ]
        )
        older_avg = statistics.mean(
            [
                s.retention_percentage
                for s in older_half
                if s.retention_percentage is not None
            ]
        )

        if recent_avg > older_avg + 5:  # 5% improvement threshold
            return "improving"
        elif recent_avg < older_avg - 5:  # 5% decline threshold
            return "declining"
        else:
            return "stable"

    # =========================================================================
    # USER-FACING METHODS
    # =========================================================================

    def get_study_statistics(self) -> Dict:
        """Get user-friendly study statistics"""
        total_sessions = self.review_sessions.count()

        if total_sessions == 0:
            return {
                "total_study_time": "0 minutes",
                "average_session_time": "0 minutes",
                "success_rate": 0.0,
                "total_sessions": 0,
                "current_streak": 0,
                "best_streak": 0,
                "mastery_level": self.mastery_level,
                "last_studied": "Never",
            }

        # Calculate success rate (Good or better)
        successful = self.review_sessions.filter(
            ReviewSession.rating >= ReviewRating.GOOD
        ).count()
        success_rate = (successful / total_sessions) * 100

        return {
            "total_study_time": f"{self.total_study_time_minutes} minutes",
            "average_session_time": f"{self.total_study_time_minutes // total_sessions} minutes",
            "success_rate": round(success_rate, 1),
            "total_sessions": total_sessions,
            "current_streak": self.current_streak,
            "best_streak": self.best_streak,
            "mastery_level": self.mastery_level.title(),
            "last_studied": (
                self.last_reviewed_date.strftime("%B %d, %Y")
                if self.last_reviewed_date
                else "Never"
            ),
        }

    def get_recent_performance(self, limit: int = 10) -> List[Dict]:
        """Get recent performance history for user display"""
        sessions = (
            self.review_sessions.order_by(ReviewSession.reviewed_at.desc())
            .limit(limit)
            .all()
        )

        performance_data = []
        for session in sessions:
            performance_data.append(
                {
                    "date": session.reviewed_at.strftime("%B %d, %Y"),
                    "time": session.reviewed_at.strftime("%I:%M %p"),
                    "performance": self._rating_to_user_friendly(session.rating),
                    "retention_rate": (
                        f"{session.retention_percentage}%"
                        if session.retention_percentage
                        else "N/A"
                    ),
                    "study_duration": (
                        f"{session.duration_minutes} min"
                        if session.duration_minutes
                        else "N/A"
                    ),
                    "response_time": (
                        f"{session.response_time_seconds:.1f}s"
                        if session.response_time_seconds
                        else "N/A"
                    ),
                    "difficulty_felt": session.difficulty_rating or "Not rated",
                }
            )

        return performance_data

    def get_progress_insights(self) -> Dict:
        """Get actionable insights about learning progress"""
        scheduler = ComprehensiveTopicScheduler()
        memory = self.to_algorithm_memory()
        strength = scheduler.calculate_realistic_topic_strength(memory)

        insights = {
            "current_strength": f"{strength['exam_adjusted_retrievability']*100:.0f}%",
            "confidence_range": f"{strength['confidence_interval']['lower']*100:.0f}% - {strength['confidence_interval']['upper']*100:.0f}%",
            "next_review": self._format_next_review_timing(),
            "study_recommendation": self._generate_study_recommendation(strength),
            "weak_areas": self._identify_weak_areas(),
            "strengths": self._identify_strengths(),
        }

        return insights

    def _rating_to_user_friendly(self, rating: int) -> str:
        """Convert numeric rating to user-friendly description"""
        descriptions = {
            ReviewRating.BLACKOUT: "Completely forgot",
            ReviewRating.AGAIN: "Struggled to recall",
            ReviewRating.HARD: "Recalled with difficulty",
            ReviewRating.GOOD: "Recalled normally",
            ReviewRating.EASY: "Knew immediately",
        }
        return descriptions.get(ReviewRating(rating), "Unknown")

    def _format_next_review_timing(self) -> str:
        """Format next review timing in user-friendly way"""
        if not self.next_review_date:
            return "Not scheduled"

        now = now_ist()
        next_review_aware = ensure_timezone_aware(self.next_review_date)
        delta = next_review_aware - now

        # Check if it's overdue
        if delta.total_seconds() < 0:
            days_overdue = abs(delta.days)
            if days_overdue == 0:
                hours_overdue = abs(delta.total_seconds()) // 3600
                return f"Overdue by {int(hours_overdue)} hours"
            else:
                return (
                    f"Overdue by {days_overdue} day{'s' if days_overdue != 1 else ''}"
                )

        # Get today's date and tomorrow's date
        today = now.date()
        tomorrow = today + timedelta(days=1)
        review_date = next_review_aware.date()

        #  Check if it's due today (same calendar date)
        if review_date == today:
            hours_until = delta.total_seconds() // 3600
            if hours_until < 1:
                minutes_until = delta.total_seconds() // 60
                return f"Due in {int(minutes_until)} minutes"
            else:
                return f"Due in {int(hours_until)} hours (today)"

        # Check if it's due tomorrow (next calendar date)
        elif review_date == tomorrow:
            hours_until = delta.total_seconds() // 3600
            return f"Due in {int(hours_until)} hours (tomorrow)"

        # Check if it's within the next week
        elif delta.days <= 7:
            day_name = next_review_aware.strftime("%A")
            return f"Due on {day_name} ({delta.days} days)"

        # Check if it's within the next month
        elif delta.days <= 30:
            return f"Due in {delta.days} days"

        # Long term
        else:
            months = delta.days // 30
            return f"Due in {months} month{'s' if months != 1 else ''}"

    def _generate_study_recommendation(self, strength: Dict) -> str:
        """Generate actionable study recommendation"""
        retrievability = strength["exam_adjusted_retrievability"]
        maturity = strength["maturity_score"]

        if retrievability < 0.5:
            return "Needs intensive review - consider breaking into smaller parts"
        elif retrievability < 0.7:
            return "Schedule extra practice sessions this week"
        elif maturity < 0.5:
            return "Keep reviewing regularly to build long-term retention"
        elif self.recent_performance_trend == "declining":
            return "Performance declining - review fundamentals"
        else:
            return "On track - continue current study schedule"

    def _identify_weak_areas(self) -> List[str]:
        """Identify areas needing improvement"""
        weak_areas = []

        if self.average_response_time and self.average_response_time > 8.0:
            weak_areas.append("Response time (taking too long to recall)")

        if self.current_streak < 3:
            weak_areas.append("Consistency")

        if self.times_forgotten > self.total_reviews * 0.3:
            weak_areas.append("Retention (forgetting too often)")

        if self.recent_performance_trend == "declining":
            weak_areas.append("Recent performance (getting worse over time)")

        return weak_areas or ["No major weak areas identified"]

    def _identify_strengths(self) -> List[str]:
        """Identify learning strengths"""
        strengths = []

        if self.current_streak >= 5:
            strengths.append(
                f"Excellent consistency ({self.current_streak} correct in a row)"
            )

        if self.average_response_time and self.average_response_time < 3.0:
            strengths.append("Fast recall (quick response times)")

        if (
            self.total_reviews > 20
            and self.successful_reviews / self.total_reviews > 0.8
        ):
            strengths.append("High success rate (getting most reviews correct)")

        if self.recent_performance_trend == "improving":
            strengths.append("Improving over time")

        return strengths or ["Building foundation - keep practicing!"]


class ReviewSession(db.Model):
    """
    Individual review session with both algorithm data and user-meaningful metrics
    """

    __tablename__ = "review_sessions"

    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(
        db.Integer, db.ForeignKey("topics.id"), nullable=False, index=True
    )

    # User-facing review data
    reviewed_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    rating = db.Column(db.Integer, nullable=False)  # ReviewRating enum
    retention_percentage = db.Column(db.Float, nullable=True)  # 0-100, user estimated
    duration_minutes = db.Column(db.Integer, nullable=True)
    response_time_seconds = db.Column(db.Float, nullable=True)
    difficulty_rating = db.Column(db.Integer, nullable=True)  # 1-10, how hard it felt

    # User notes and context
    notes = db.Column(db.Text, nullable=True)
    study_method = db.Column(
        db.String(100), nullable=True
    )  # flashcards, reading, practice, etc.
    environment = db.Column(db.String(100), nullable=True)  # home, library, etc.

    # Algorithm metadata (hidden from user)
    stability_before = db.Column(db.Float, nullable=True)
    stability_after = db.Column(db.Float, nullable=True)
    difficulty_before = db.Column(db.Float, nullable=True)
    difficulty_after = db.Column(db.Float, nullable=True)
    predicted_success_rate = db.Column(db.Float, nullable=True)

    # Session context
    session_number = db.Column(db.Integer, nullable=True)  # Nth review of this topic
    days_since_last_review = db.Column(db.Float, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __init__(self, topic_id: int, rating: int, **kwargs):
        super().__init__()
        print('I am called')
        self.topic_id = topic_id
        self.rating = rating
        self.reviewed_at = kwargs.get("reviewed_at", now_ist())
        

        # Set other attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def to_user_display(self) -> Dict:
        """Convert to user-friendly display format"""
        return {
            "date": self.reviewed_at.strftime("%B %d, %Y at %I:%M %p"),
            "performance": self._get_performance_description(),
            "retention_rate": (
                f"{self.retention_percentage}%"
                if self.retention_percentage
                else "Not estimated"
            ),
            "time_spent": (
                f"{self.duration_minutes} minutes"
                if self.duration_minutes
                else "Not recorded"
            ),
            "response_speed": self._get_response_speed_description(),
            "difficulty_felt": (
                f"{self.difficulty_rating}/10"
                if self.difficulty_rating
                else "Not rated"
            ),
            "notes": self.notes or "No notes recorded",
            "study_method": self.study_method or "Not specified",
        }

    def _get_performance_description(self) -> str:
        """Get user-friendly performance description"""
        descriptions = {
            ReviewRating.BLACKOUT: ("Complete blank", "🔴"),
            ReviewRating.AGAIN: ("Struggled", "🟡"),
            ReviewRating.HARD: ("Difficult but got it", "🟠"),
            ReviewRating.GOOD: ("Recalled well", "🟢"),
            ReviewRating.EASY: ("Knew immediately", "✅"),
        }
        desc, emoji = descriptions.get(ReviewRating(self.rating), ("Unknown", ""))
        return f"{emoji} {desc}"

    def _get_response_speed_description(self) -> str:
        """Get user-friendly response speed description"""
        if not self.response_time_seconds:
            return "Not measured"

        if self.response_time_seconds < 2:
            return f"Very fast ({self.response_time_seconds:.1f}s)"
        elif self.response_time_seconds < 5:
            return f"Normal speed ({self.response_time_seconds:.1f}s)"
        elif self.response_time_seconds < 10:
            return f"Slow ({self.response_time_seconds:.1f}s)"
        else:
            return f"Very slow ({self.response_time_seconds:.1f}s)"


class Exam(db.Model):
    """
    Exam with student-focused information and preparation tracking
    """

    __tablename__ = "exams"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    exam_date = db.Column(db.Date, nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)

    # User-meaningful exam information
    importance = db.Column(
        db.String(20), default="medium", nullable=False
    )  # low/medium/high/critical
    exam_type = db.Column(
        db.String(50), nullable=True
    )  # final, midterm, quiz, certification
    total_marks = db.Column(db.Integer, nullable=True)
    passing_marks = db.Column(db.Integer, nullable=True)

    # Preparation tracking
    preparation_status = db.Column(
        db.String(20), default="not_started", nullable=False
    )  # not_started/in_progress/well_prepared/review_ready
    estimated_study_hours_needed = db.Column(db.Float, nullable=True)
    actual_study_hours = db.Column(db.Float, default=0.0, nullable=False)

    # Auto-calculated metrics (updated by algorithm)
    overall_readiness_score = db.Column(db.Float, default=0.0, nullable=False)  # 0-100
    topics_ready_count = db.Column(db.Integer, default=0, nullable=False)
    topics_total_count = db.Column(db.Integer, default=0, nullable=False)
    estimated_performance = db.Column(
        db.String(20), nullable=True
    )  # excellent/good/fair/poor/critical

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    topic_associations = db.relationship(
        "ExamTopic", backref="exam", lazy="dynamic", cascade="all, delete-orphan"
    )
    study_plans = db.relationship(
        "StudyPlan", backref="exam", lazy="dynamic", cascade="all, delete-orphan"
    )

    def get_preparation_summary(self) -> Dict:
        """Get comprehensive preparation summary for student"""
        topics = [assoc.topic for assoc in self.topic_associations]
        days_remaining = (self.exam_date - date.today()).days

        # Calculate readiness by topic
        topic_readiness = []
        total_readiness = 0
        ready_count = 0

        scheduler = ComprehensiveTopicScheduler()

        for topic in topics:
            memory = topic.to_algorithm_memory()
            strength = scheduler.calculate_realistic_topic_strength(
                memory,
                exam_context={
                    "overall_preparation": self.overall_readiness_score / 100
                },
            )

            readiness_score = strength["exam_adjusted_retrievability"]
            total_readiness += readiness_score

            if readiness_score >= 0.8:  # 80% threshold for "ready"
                ready_count += 1

            topic_readiness.append(
                {
                    "name": topic.name,
                    "subject": topic.subject,
                    "readiness_score": round(readiness_score * 100, 1),
                    "readiness_level": strength["readiness_category"],
                    "confidence_range": f"{strength['confidence_interval']['lower']*100:.0f}%-{strength['confidence_interval']['upper']*100:.0f}%",
                    "last_studied": (
                        topic.last_reviewed_date.strftime("%b %d")
                        if topic.last_reviewed_date
                        else "Never"
                    ),
                    "next_review": topic._format_next_review_timing(),
                    "mastery_level": topic.mastery_level.title(),
                }
            )

        # Overall statistics
        avg_readiness = (total_readiness / len(topics) * 100) if topics else 0

        # Generate preparation status
        if days_remaining <= 0:
            status_message = "Exam has passed"
            urgency_level = "completed"
        elif avg_readiness >= 85:
            status_message = "Excellent preparation! You're ready."
            urgency_level = "ready"
        elif avg_readiness >= 70:
            status_message = "Good preparation with some areas to review"
            urgency_level = "mostly_ready"
        elif avg_readiness >= 55:
            status_message = "Moderate preparation - focus on weak topics"
            urgency_level = "needs_work"
        else:
            status_message = "Intensive study needed immediately"
            urgency_level = "critical"

        # Time management
        if days_remaining > 0:
            recommended_daily_hours = max(
                1.0,
                min(6.0, (self.estimated_study_hours_needed or 20) / days_remaining),
            )
        else:
            recommended_daily_hours = 0

        return {
            "exam_name": self.name,
            "exam_date": self.exam_date.strftime("%B %d, %Y"),
            "days_remaining": days_remaining,
            "overall_readiness": round(avg_readiness, 1),
            "topics_ready": ready_count,
            "total_topics": len(topics),
            "status_message": status_message,
            "urgency_level": urgency_level,
            "topic_breakdown": sorted(
                topic_readiness, key=lambda x: x["readiness_score"]
            ),
            "study_recommendations": {
                "daily_hours_needed": round(recommended_daily_hours, 1),
                "total_hours_remaining": round(
                    (self.estimated_study_hours_needed or 0) - self.actual_study_hours,
                    1,
                ),
                "priority_topics": [
                    t["name"]
                    for t in sorted(
                        topic_readiness, key=lambda x: x["readiness_score"]
                    )[:3]
                ],
                "focus_areas": self._get_focus_recommendations(topic_readiness),
            },
        }

    def _get_focus_recommendations(self, topic_readiness: List[Dict]) -> List[str]:
        """Generate specific focus recommendations"""
        recommendations = []

        weak_topics = [t for t in topic_readiness if t["readiness_score"] < 60]
        if len(weak_topics) > 5:
            recommendations.append(
                f"Too many weak topics ({len(weak_topics)}) - consider prioritizing most important ones"
            )
        elif weak_topics:
            recommendations.append(
                f"Focus on {len(weak_topics)} topics scoring below 60%"
            )

        never_studied = [t for t in topic_readiness if t["last_studied"] == "Never"]
        if never_studied:
            recommendations.append(
                f"Start with {len(never_studied)} topics you haven't studied yet"
            )

        overdue_topics = [t for t in topic_readiness if "Overdue" in t["next_review"]]
        if overdue_topics:
            recommendations.append(f"Catch up on {len(overdue_topics)} overdue reviews")

        return recommendations or ["Maintain current study schedule"]


class ExamTopic(db.Model):
    """
    Association between topics and exams with weighting
    """

    __tablename__ = "exam_topics"

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.id"), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey("topics.id"), nullable=False)

    # User-meaningful weighting information
    importance_weight = db.Column(
        db.Float, default=1.0, nullable=False
    )  # How important for this exam
    expected_marks_percentage = db.Column(
        db.Float, nullable=True
    )  # What % of exam marks
    confidence_level = db.Column(
        db.String(20), default="medium", nullable=False
    )  # low/medium/high

    # Study planning
    estimated_study_hours = db.Column(db.Float, nullable=True)
    actual_study_hours = db.Column(db.Float, default=0.0, nullable=False)
    priority_rank = db.Column(db.Integer, nullable=True)  # 1=highest priority

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Unique constraint
    __table_args__ = (db.UniqueConstraint("exam_id", "topic_id"),)


class StudyPlan(db.Model):
    """
    Generated study plans for exam preparation
    """

    __tablename__ = "study_plans"

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exams.id"), nullable=False)

    # Plan metadata
    plan_name = db.Column(db.String(200), nullable=False)
    created_date = db.Column(db.Date, nullable=False, default=date.today)
    target_exam_date = db.Column(db.Date, nullable=False)
    total_study_days = db.Column(db.Integer, nullable=False)
    daily_study_hours = db.Column(db.Float, nullable=False)

    # Progress tracking
    completed_sessions = db.Column(db.Integer, default=0, nullable=False)
    total_planned_sessions = db.Column(db.Integer, nullable=False)
    adherence_percentage = db.Column(db.Float, default=0.0, nullable=False)

    # Plan details (JSON)
    daily_schedule = db.Column(db.Text, nullable=False)  # JSON array of daily plans

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def get_today_plan(self) -> Dict:
        """Get today's study plan"""
        try:
            schedule = json.loads(self.daily_schedule)
            today_str = date.today().strftime("%Y-%m-%d")

            for day_plan in schedule:
                if day_plan.get("date") == today_str:
                    return day_plan

            return {"message": "No plan for today", "topics": [], "estimated_hours": 0}
        except (json.JSONDecodeError, KeyError):
            return {
                "message": "Plan data corrupted",
                "topics": [],
                "estimated_hours": 0,
            }

    def get_weekly_overview(self) -> Dict:
        """Get this week's study overview"""
        try:
            schedule = json.loads(self.daily_schedule)
            today = date.today()

            # Get current week (Monday to Sunday)
            week_start = today - timedelta(days=today.weekday())
            week_dates = [
                (week_start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)
            ]

            week_plan = []
            total_hours = 0
            completed_days = 0

            for date_str in week_dates:
                day_plan = next(
                    (d for d in schedule if d.get("date") == date_str), None
                )
                if day_plan:
                    week_plan.append(day_plan)
                    total_hours += day_plan.get("estimated_hours", 0)
                    if day_plan.get("completed", False):
                        completed_days += 1

            return {
                "week_start": week_start.strftime("%B %d"),
                "week_end": (week_start + timedelta(days=6)).strftime("%B %d"),
                "daily_plans": week_plan,
                "total_hours_planned": total_hours,
                "days_completed": completed_days,
                "completion_rate": (
                    round((completed_days / 7) * 100, 1) if week_plan else 0
                ),
            }
        except (json.JSONDecodeError, KeyError):
            return {"error": "Unable to load weekly plan"}


class StudySession(db.Model):
    """
    Actual study sessions (different from reviews - broader study activities)
    """

    __tablename__ = "study_sessions"

    id = db.Column(db.Integer, primary_key=True)

    # Session details
    started_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)
    duration_minutes = db.Column(db.Integer, nullable=True)

    # Content studied
    topics_covered = db.Column(db.Text, nullable=True)  # JSON array of topic IDs
    study_method = db.Column(
        db.String(100), nullable=True
    )  # reading, practice_problems, flashcards, etc.
    materials_used = db.Column(db.Text, nullable=True)  # textbooks, videos, notes, etc.

    # User reflection
    productivity_rating = db.Column(db.Integer, nullable=True)  # 1-10 how productive
    difficulty_rating = db.Column(db.Integer, nullable=True)  # 1-10 how difficult
    fatigue_level_start = db.Column(db.Integer, nullable=True)  # 1-10
    fatigue_level_end = db.Column(db.Integer, nullable=True)  # 1-10

    # Environment and context
    location = db.Column(db.String(100), nullable=True)
    distractions = db.Column(db.String(200), nullable=True)
    mood_before = db.Column(db.String(50), nullable=True)
    mood_after = db.Column(db.String(50), nullable=True)

    # Notes and insights
    session_notes = db.Column(db.Text, nullable=True)
    insights_learned = db.Column(db.Text, nullable=True)
    challenges_faced = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def get_session_summary(self) -> Dict:
        """Get user-friendly session summary"""
        if not self.ended_at:
            return {"status": "in_progress", "duration": "Ongoing"}

        duration = self.duration_minutes or 0
        topics_list = []

        try:
            if self.topics_covered:
                topic_ids = json.loads(self.topics_covered)
                topics_list = [
                    Topic.query.get(tid).name
                    for tid in topic_ids
                    if Topic.query.get(tid)
                ]
        except (json.JSONDecodeError, AttributeError):
            topics_list = []

        return {
            "date": self.started_at.strftime("%B %d, %Y"),
            "duration": f"{duration} minutes",
            "topics_studied": topics_list,
            "study_method": self.study_method or "Not specified",
            "productivity": (
                f"{self.productivity_rating}/10"
                if self.productivity_rating
                else "Not rated"
            ),
            "key_insights": self.insights_learned or "No insights recorded",
            "challenges": self.challenges_faced or "No challenges noted",
        }


class LearningAnalytics(db.Model):
    """
    Aggregated learning analytics and insights (computed periodically)
    """

    __tablename__ = "learning_analytics"

    id = db.Column(db.Integer, primary_key=True)

    # Time period for analytics
    period_start = db.Column(db.Date, nullable=False, index=True)
    period_end = db.Column(db.Date, nullable=False)
    period_type = db.Column(db.String(20), nullable=False)  # daily, weekly, monthly

    # Learning metrics
    total_study_time = db.Column(db.Integer, default=0, nullable=False)  # minutes
    total_reviews = db.Column(db.Integer, default=0, nullable=False)
    successful_reviews = db.Column(db.Integer, default=0, nullable=False)
    topics_studied = db.Column(db.Integer, default=0, nullable=False)
    new_topics_learned = db.Column(db.Integer, default=0, nullable=False)

    # Performance metrics
    average_retention_rate = db.Column(db.Float, nullable=True)
    average_response_time = db.Column(db.Float, nullable=True)
    consistency_score = db.Column(
        db.Float, nullable=True
    )  # How consistent study schedule was

    # Behavioral insights
    most_productive_time = db.Column(
        db.String(20), nullable=True
    )  # morning, afternoon, evening
    preferred_study_methods = db.Column(db.Text, nullable=True)  # JSON array
    common_challenge_areas = db.Column(db.Text, nullable=True)  # JSON array

    # Goal tracking
    study_time_goal = db.Column(db.Integer, nullable=True)  # minutes
    review_goal = db.Column(db.Integer, nullable=True)
    goal_achievement_rate = db.Column(db.Float, nullable=True)  # 0.0 to 1.0

    computed_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    @staticmethod
    def generate_weekly_analytics(week_start_date: date) -> "LearningAnalytics":
        """Generate analytics for a specific week"""
        week_end = week_start_date + timedelta(days=6)

        # Get all review sessions in this week
        week_reviews = ReviewSession.query.filter(
            ReviewSession.reviewed_at
            >= datetime.combine(week_start_date, datetime.min.time()),
            ReviewSession.reviewed_at
            <= datetime.combine(week_end, datetime.max.time()),
        ).all()

        # Get all study sessions in this week
        week_studies = StudySession.query.filter(
            StudySession.started_at
            >= datetime.combine(week_start_date, datetime.min.time()),
            StudySession.started_at <= datetime.combine(week_end, datetime.max.time()),
        ).all()

        # Calculate metrics
        total_reviews = len(week_reviews)
        successful_reviews = len(
            [r for r in week_reviews if r.rating >= ReviewRating.GOOD]
        )
        total_study_time = sum([s.duration_minutes or 0 for s in week_studies])

        # Calculate average retention
        retention_rates = [
            r.retention_percentage
            for r in week_reviews
            if r.retention_percentage is not None
        ]
        avg_retention = statistics.mean(retention_rates) if retention_rates else None

        # Calculate response times
        response_times = [
            r.response_time_seconds
            for r in week_reviews
            if r.response_time_seconds is not None
        ]
        avg_response_time = statistics.mean(response_times) if response_times else None

        # Count unique topics
        topics_studied = len(set(r.topic_id for r in week_reviews))

        analytics = LearningAnalytics(
            period_start=week_start_date,
            period_end=week_end,
            period_type="weekly",
            total_study_time=total_study_time,
            total_reviews=total_reviews,
            successful_reviews=successful_reviews,
            topics_studied=topics_studied,
            average_retention_rate=avg_retention,
            average_response_time=avg_response_time,
        )

        return analytics

    def get_insights_summary(self) -> Dict:
        """Get user-friendly insights from analytics"""
        success_rate = (self.successful_reviews / max(1, self.total_reviews)) * 100

        # Generate insights based on data
        insights = []

        if success_rate >= 80:
            insights.append(
                "Excellent performance! You're mastering topics consistently."
            )
        elif success_rate >= 60:
            insights.append("Good progress, but some topics need more attention.")
        else:
            insights.append("Focus on understanding - success rate could be improved.")

        if self.total_study_time < 60:  # Less than 1 hour per day average
            insights.append("Consider increasing daily study time for better results.")
        elif self.total_study_time > 300:  # More than 5 hours per day average
            insights.append(
                "High study volume - make sure to take breaks to avoid burnout."
            )

        if self.average_response_time and self.average_response_time > 8:
            insights.append("Response times are slow - focus on quick recall practice.")

        return {
            "period": f"{self.period_start.strftime('%b %d')} - {self.period_end.strftime('%b %d')}",
            "total_study_hours": round(self.total_study_time / 60, 1),
            "success_rate": round(success_rate, 1),
            "topics_covered": self.topics_studied,
            "average_retention": (
                round(self.average_retention_rate, 1)
                if self.average_retention_rate
                else "N/A"
            ),
            "key_insights": insights,
        }


# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================


def initialize_database():
    """Initialize database with proper indexes and constraints"""
    db.create_all()

    # Create additional indexes for performance
    db.engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_topics_next_review ON topics(next_review_date);"
    )
    db.engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_topics_subject_mastery ON topics(subject, mastery_level);"
    )
    db.engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_reviews_date_rating ON review_sessions(reviewed_at, rating);"
    )
    db.engine.execute(
        "CREATE INDEX IF NOT EXISTS idx_exams_date_status ON exams(exam_date, preparation_status);"
    )


def get_due_topics(limit: int = 20) -> List[Topic]:
    """Get topics due for review"""
    current_time = now_ist()
    return (
        Topic.query.filter(Topic.next_review_date <= current_time)
        .order_by(Topic.next_review_date.asc())
        .limit(limit)
        .all()
    )


def get_upcoming_exams(days_ahead: int = 30) -> List[Exam]:
    """Get exams coming up within specified days"""
    cutoff_date = date.today() + timedelta(days=days_ahead)
    return (
        Exam.query.filter(Exam.exam_date >= date.today(), Exam.exam_date <= cutoff_date)
        .order_by(Exam.exam_date.asc())
        .all()
    )


def process_topic_review(topic_id: int, rating: int, **kwargs) -> Dict:
    """Process a topic review using the algorithm"""
    topic = Topic.query.get_or_404(topic_id)

    # Create review session record
    review_session = ReviewSession(topic_id=topic_id, rating=rating, **kwargs)

    # Process with algorithm
    scheduler = ComprehensiveTopicScheduler()
    memory = topic.to_algorithm_memory()

    # Get algorithm result
    result = scheduler.process_review(
        memory,
        Rating(rating),
        response_time_seconds=kwargs.get("response_time_seconds"),
        study_context=kwargs.get("study_context", {}),
    )

    # Update topic from algorithm result
    topic.update_from_algorithm_result(result)

    # Update review session with algorithm metadata
    review_session.stability_before = memory.stability
    review_session.stability_after = result["updated_memory"].stability
    review_session.difficulty_before = memory.difficulty
    review_session.difficulty_after = result["updated_memory"].difficulty
    review_session.predicted_success_rate = result["pre_review_retrievability"]
    review_session.session_number = topic.total_reviews

    if memory.last_review_date:
        days_since = (datetime.now() - memory.last_review_date).days
        review_session.days_since_last_review = days_since

    # Update topic's total study time
    if kwargs.get("duration_minutes"):
        topic.total_study_time_minutes += kwargs["duration_minutes"]

    # Save to database
    db.session.add(review_session)
    db.session.commit()

    return {
        "success": True,
        "topic_updated": topic.to_dict() if hasattr(topic, "to_dict") else str(topic),
        "review_session": review_session.to_user_display(),
        "algorithm_result": result,
        "next_review": topic._format_next_review_timing(),
    }


def get_learning_dashboard() -> dict:
    """Get comprehensive learning dashboard data"""
    # Due topics
    due_topics = get_due_topics(10)

    # Upcoming exams
    upcoming_exams = get_upcoming_exams(30)

    # This week's activity
    from utils.datetime_utils import get_this_week_s_monday
    print(now_ist())
    week_ago = get_this_week_s_monday()
        
    recent_reviews = ReviewSession.query.filter(
        ReviewSession.reviewed_at >= week_ago
    ).count()

    week_start = date.today() - timedelta(days=date.today().weekday())
    week_end = week_start + timedelta(days=6)

    recent_study_time = (
        db.session.query(
            db.func.coalesce(db.func.sum(ReviewSession.duration_minutes), 0)
        )
        .filter(
            ReviewSession.reviewed_at
            >= datetime.combine(week_start, datetime.min.time()),
            ReviewSession.reviewed_at
            <= datetime.combine(week_end, datetime.max.time()),
        )
        .scalar()
    )

    # Learning streaks
    current_streak = 0
    check_date = date.today()
    while True:
        day_reviews = ReviewSession.query.filter(
            db.func.date(ReviewSession.reviewed_at) == check_date
        ).count()

        if day_reviews == 0:
            break
        current_streak += 1
        check_date -= timedelta(days=1)

    # Performance insights
    scheduler = ComprehensiveTopicScheduler()
    all_topics = Topic.query.all()

    readiness_distribution = {
        "excellent": 0,
        "good": 0,
        "fair": 0,
        "poor": 0,
        "critical": 0,
    }

    confidence_scores = []

    for topic in all_topics:
        memory = topic.to_algorithm_memory()
        strength = scheduler.calculate_realistic_topic_strength(memory)
        readiness_distribution[strength["readiness_category"]] += 1

        # Use real-time prediction confidence
        confidence_scores.append(strength["prediction_confidence"])
    avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
    return {
        "due_for_review": {
            "count": len(due_topics),
            "topics": [
                {
                    "name": t.name,
                    "subject": t.subject,
                    "overdue_by": (datetime.now() - t.next_review_date).days,
                }
                for t in due_topics[:5]
            ],
        },
        "upcoming_exams": {
            "count": len(upcoming_exams),
            "exams": [
                {
                    "name": e.name,
                    "date": e.exam_date.strftime("%B %d"),
                    "days_left": (e.exam_date - date.today()).days,
                    "readiness": f"{e.overall_readiness_score}%",
                }
                for e in upcoming_exams[:3]
            ],
        },
        "recent_activity": {
            "reviews_completed": recent_reviews,
            "study_hours": round(recent_study_time / 60, 1),
            "current_streak": current_streak,
        },
        "knowledge_overview": {
            "total_topics": len(all_topics),
            "mastery_distribution": readiness_distribution,
            "avg_confidence": round(avg_confidence, 2) if avg_confidence else 0,
        },
    }


# Usage example for Flask routes:
"""
@app.route('/api/review/<int:topic_id>', methods=['POST'])
def review_topic(topic_id):
    data = request.json
    result = process_topic_review(
        topic_id=topic_id,
        rating=data['rating'],
        retention_percentage=data.get('retention_percentage'),
        duration_minutes=data.get('duration_minutes'),
        response_time_seconds=data.get('response_time_seconds'),
        difficulty_rating=data.get('difficulty_rating'),
        notes=data.get('notes')
    )
    return jsonify(result)

@app.route('/api/dashboard')
def dashboard():
    data = get_learning_dashboard()
    return jsonify(data)
"""
