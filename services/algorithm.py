"""
Robust FSRS-based Algorithm System for Topic Learning
=====================================================

This module implements a comprehensive spaced repetition algorithm specifically
designed for exam preparation and topic mastery. Unlike simplified implementations,
this version models the harsh realities of human memory, cognitive limitations,
and exam pressure.

Key Features:
- Realistic forgetting curves with interference modeling
- Cognitive load management and burnout prevention
- Exam pressure simulation and stress impact
- Adaptive parameter learning from user performance
- Comprehensive performance validation and confidence intervals
- Study session optimization and workload balancing

Architecture:
- Core FSRS mathematical foundation (research-backed)
- Layered systems for different aspects of learning
- Extensive validation and reality checks
- Conservative estimates with confidence bounds
"""

import math
import statistics
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import IntEnum
import random
from collections import defaultdict

# Core Constants (Based on FSRS Research)
FSRS_DEFAULT_DECAY = -0.1542
DEFAULT_PARAMETERS = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
]

# Realistic Learning Constraints
MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
STABILITY_MIN = 0.1
STABILITY_MAX = 36500  # 100 years theoretical max
FORGETTING_FLOOR = 0.05  # You never forget everything completely
RETRIEVAL_CEILING = 0.98  # Perfect recall is impossible
DAILY_REVIEW_CAPACITY_BASE = 50  # Conservative estimate
COGNITIVE_INTERFERENCE_THRESHOLD = 10  # Topics before interference kicks in
CRAMMING_EFFECTIVENESS_DECAY = 0.7  # How much cramming helps vs spaced learning


class TopicState(IntEnum):
    """Learning states with realistic transitions"""

    NEW = 0  # Never studied
    LEARNING = 1  # Initial acquisition phase
    YOUNG = 2  # Recently learned, not yet stable
    REVIEW = 3  # Stable knowledge, periodic review
    MATURE = 4  # Well-consolidated, long intervals
    RELEARNING = 5  # Forgotten, needs reacquisition
    SUSPENDED = 6  # Temporarily disabled


class Rating(IntEnum):
    """Performance ratings with clear criteria"""

    BLACKOUT = 1  # Complete failure, no recognition
    AGAIN = 2  # Recognized but couldn't recall
    HARD = 3  # Recalled with significant effort
    GOOD = 4  # Recalled normally
    EASY = 5  # Effortless, immediate recall


class LearningPhase(IntEnum):
    """Different phases of learning process"""

    ACQUISITION = 1  # Initial learning
    CONSOLIDATION = 2  # Strengthening memory
    MAINTENANCE = 3  # Long-term retention
    REACQUISITION = 4  # Recovering forgotten knowledge


@dataclass
class StudySession:
    """Represents a single study/review session"""

    timestamp: datetime
    duration_minutes: int
    topics_reviewed: List[int]  # Topic IDs
    average_difficulty: float
    fatigue_level: float  # 0.0 to 1.0
    context_switches: int
    performance_scores: List[float]


@dataclass
class CognitiveState:
    """Current cognitive capacity and fatigue"""

    current_capacity: float  # 0.0 to 1.0
    daily_reviews_completed: int
    consecutive_study_days: int
    last_break_hours: float
    stress_level: float  # 0.0 to 1.0
    motivation_level: float  # 0.0 to 1.0


@dataclass
class TopicMemory:
    """Complete memory representation of a topic"""

    # Core FSRS parameters
    stability: float
    difficulty: float

    # Memory characteristics
    initial_learning_date: Optional[datetime]
    last_review_date: Optional[datetime]
    due_date: datetime
    state: TopicState

    # Learning history
    review_count: int
    lapses: int  # Times forgotten
    streak: int  # Successful reviews in a row

    # Performance tracking
    average_retention: float
    response_times: List[float]  # Historical response times
    recent_ratings: List[Rating]  # Last 10 ratings

    # Context and interference
    similar_topics: List[int]  # Related topic IDs
    subject_category: Optional[str]
    complexity_score: float  # 1.0 to 10.0

    # Exam context
    exam_dates: List[date]
    exam_weights: List[float]
    cramming_sessions: int  # Count of cramming vs spaced reviews


class CoreFSRSEngine:
    """Mathematical foundation - implements research-backed FSRS algorithm"""

    def __init__(self, parameters: List[float] = None):
        self.parameters = parameters or DEFAULT_PARAMETERS.copy()
        self._decay = -self.parameters[20]
        self._factor = 0.9 ** (1 / self._decay) - 1

    def calculate_initial_stability(self, rating: Rating) -> float:
        """S₀(G) = w[G-1], constrained to realistic bounds"""
        if rating == Rating.BLACKOUT:
            return STABILITY_MIN

        raw_stability = self.parameters[rating - 2]  # Adjust for enum offset
        return max(STABILITY_MIN, min(raw_stability, 7.0))  # Cap initial at 1 week

    def calculate_initial_difficulty(self, rating: Rating) -> float:
        """D₀(G) = w₄ - e^((G-1) × w₅) + 1, with realistic constraints"""
        if rating == Rating.BLACKOUT:
            return MAX_DIFFICULTY

        raw_difficulty = (
            self.parameters[4] - math.exp(self.parameters[5] * (rating - 2)) + 1
        )
        return max(MIN_DIFFICULTY, min(raw_difficulty, MAX_DIFFICULTY))

    def calculate_retrievability(self, stability: float, days_elapsed: float) -> float:
        """R(t,S) = (1 + FACTOR × t/(9×S))^DECAY with reality constraints"""
        if days_elapsed <= 0:
            return 0.98  # High but not perfect for immediate review

        # Core FSRS formula
        retrievability = (
            1 + self._factor * days_elapsed / (9 * stability)
        ) ** self._decay

        # Apply reality constraints
        retrievability = max(FORGETTING_FLOOR, min(RETRIEVAL_CEILING, retrievability))

        return retrievability

    def calculate_next_stability_success(
        self,
        current_stability: float,
        difficulty: float,
        retrievability: float,
        rating: Rating,
    ) -> float:
        """Calculate stability after successful recall"""
        hard_penalty = self.parameters[15] if rating == Rating.HARD else 1.0
        easy_bonus = self.parameters[16] if rating == Rating.EASY else 1.0

        next_stability = current_stability * (
            1
            + math.exp(self.parameters[8])
            * (11 - difficulty)
            * (current_stability ** -self.parameters[9])
            * (math.exp((1 - retrievability) * self.parameters[10]) - 1)
            * hard_penalty
            * easy_bonus
        )

        return max(STABILITY_MIN, min(next_stability, STABILITY_MAX))

    def calculate_next_stability_failure(
        self, current_stability: float, difficulty: float, retrievability: float
    ) -> float:
        """Calculate stability after forgetting"""
        # Long-term forgetting component
        forget_stability = (
            self.parameters[11]
            * (difficulty ** -self.parameters[12])
            * ((current_stability + 1) ** self.parameters[13] - 1)
            * math.exp((1 - retrievability) * self.parameters[14])
        )

        # Short-term constraint (can't be better than short-term stability)
        short_term_limit = current_stability / math.exp(
            self.parameters[17] * self.parameters[18]
        )

        next_stability = min(forget_stability, short_term_limit)
        return max(STABILITY_MIN, min(next_stability, STABILITY_MAX))

    def calculate_next_difficulty(
        self, current_difficulty: float, rating: Rating
    ) -> float:
        """D_next = mean_reversion(w₇, D + linear_damping(Δ_D))"""
        # Difficulty change based on rating
        delta_difficulty = -self.parameters[6] * (rating - 3)

        # Linear damping function
        damped_change = (10 - current_difficulty) * delta_difficulty / 9
        intermediate_difficulty = current_difficulty + damped_change

        # Mean reversion to easy rating difficulty
        target_difficulty = self.calculate_initial_difficulty(Rating.EASY)
        next_difficulty = (
            self.parameters[7] * target_difficulty
            + (1 - self.parameters[7]) * intermediate_difficulty
        )

        return max(MIN_DIFFICULTY, min(next_difficulty, MAX_DIFFICULTY))


class MemoryStateManager:
    """Manages realistic state transitions and memory consolidation"""

    # Graduation thresholds (conservative)
    LEARNING_GRADUATION_THRESHOLD = 3  # Must get Good+ 3 times
    YOUNG_TO_REVIEW_DAYS = 21  # 3 weeks to stabilize
    REVIEW_TO_MATURE_INTERVAL = 90  # 3 months of successful reviews
    LAPSE_THRESHOLD = 0.6  # Below this retrievability = lapsed

    def __init__(self):
        self.fsrs = CoreFSRSEngine()

    def determine_next_state(self, memory: TopicMemory, rating: Rating, 
                        days_since_last: float) -> TopicState:
        """Determine state transition based on performance and history"""
        current_retrievability = self.fsrs.calculate_retrievability(
            memory.stability, days_since_last
        )
        
        # Check for lapse first (more forgiving for topics)
        if rating <= Rating.AGAIN and current_retrievability < self.LAPSE_THRESHOLD:
            return TopicState.RELEARNING

        # State-specific transitions
        if memory.state == TopicState.NEW:
            return TopicState.LEARNING

        elif memory.state == TopicState.LEARNING:
            # FIXED: Allow EASY rating to graduate immediately
            if rating == Rating.EASY and memory.review_count >= 1:
                return TopicState.YOUNG
            elif rating >= Rating.GOOD and memory.streak >= self.LEARNING_GRADUATION_THRESHOLD:
                return TopicState.YOUNG
            elif rating == Rating.HARD and memory.streak >= 1:  # Allow graduation with HARD rating
                return TopicState.YOUNG
            return TopicState.LEARNING

        elif memory.state == TopicState.YOUNG:
            if rating <= Rating.AGAIN:
                return TopicState.RELEARNING

            days_in_young = (datetime.now() - memory.initial_learning_date).days
            if days_in_young >= self.YOUNG_TO_REVIEW_DAYS and memory.streak >= 5:
                return TopicState.REVIEW
            return TopicState.YOUNG

        elif memory.state == TopicState.REVIEW:
            if rating <= Rating.AGAIN:
                return TopicState.RELEARNING

            if (
                memory.stability >= self.REVIEW_TO_MATURE_INTERVAL
                and memory.average_retention >= 0.85
                and memory.streak >= 8
            ):
                return TopicState.MATURE
            return TopicState.REVIEW

        elif memory.state == TopicState.MATURE:
            if rating <= Rating.AGAIN:
                return TopicState.RELEARNING
            return TopicState.MATURE

        elif memory.state == TopicState.RELEARNING:
            if rating >= Rating.GOOD and memory.streak >= 2:
                # Return to previous stable state
                if memory.review_count >= 20:
                    return TopicState.REVIEW
                else:
                    return TopicState.YOUNG
            elif rating == Rating.HARD and memory.streak >= 1:  # Allow HARD to count as recovery
                return TopicState.YOUNG
            return TopicState.RELEARNING

        return memory.state


    def calculate_state_interval_modifier(self, state: TopicState) -> float:
        """Adjust intervals based on consolidation state"""
        modifiers = {
            TopicState.NEW: 0.0,  # No interval yet
            TopicState.LEARNING: 0.1,  # Very short intervals
            TopicState.YOUNG: 0.5,  # Conservative intervals
            TopicState.REVIEW: 1.0,  # Standard intervals
            TopicState.MATURE: 1.5,  # Longer intervals allowed
            TopicState.RELEARNING: 0.3,  # Rebuild gradually
            TopicState.SUSPENDED: 0.0,  # No scheduling
        }
        return modifiers.get(state, 1.0)


class CognitiveLoadManager:
    """Models human cognitive limitations and fatigue"""

    def __init__(self):
        self.base_daily_capacity = DAILY_REVIEW_CAPACITY_BASE
        self.interference_threshold = COGNITIVE_INTERFERENCE_THRESHOLD
        self.fatigue_accumulation_rate = 0.02  # Per review
        self.recovery_rate = 0.1  # Per hour of rest

    def calculate_current_capacity(
        self, cognitive_state: CognitiveState, recent_sessions: List[StudySession]
    ) -> float:
        """Calculate current cognitive capacity (0.0 to 1.0)"""
        base_capacity = 1.0

        # Fatigue from recent activity
        daily_fatigue = min(
            0.7,
            cognitive_state.daily_reviews_completed * self.fatigue_accumulation_rate,
        )

        # Cumulative fatigue from consecutive days
        consecutive_fatigue = min(0.4, cognitive_state.consecutive_study_days * 0.05)

        # Recovery from breaks
        recovery_boost = min(0.3, cognitive_state.last_break_hours * self.recovery_rate)

        # Stress penalty
        stress_penalty = cognitive_state.stress_level * 0.3

        # Motivation boost
        motivation_boost = (cognitive_state.motivation_level - 0.5) * 0.2

        current_capacity = (
            base_capacity
            - daily_fatigue
            - consecutive_fatigue
            - stress_penalty
            + recovery_boost
            + motivation_boost
        )

        return max(0.1, min(1.0, current_capacity))

    def calculate_interference_penalty(
        self, similar_topics_count: int, context_switches: int
    ) -> float:
        """Calculate memory interference from similar topics"""
        # Interference grows exponentially with similar topics
        topic_interference = 1.0 - math.exp(
            -similar_topics_count / self.interference_threshold
        )

        # Context switching penalty
        switch_penalty = min(0.3, context_switches * 0.05)

        total_interference = min(0.5, topic_interference + switch_penalty)
        return 1.0 - total_interference

    def estimate_review_duration(
        self, memory: TopicMemory, cognitive_capacity: float
    ) -> float:
        """Estimate time needed for review in minutes"""
        base_time = {
            TopicState.NEW: 5.0,
            TopicState.LEARNING: 3.0,
            TopicState.YOUNG: 2.0,
            TopicState.REVIEW: 1.5,
            TopicState.MATURE: 1.0,
            TopicState.RELEARNING: 4.0,
        }.get(memory.state, 2.0)

        # Adjust for complexity and difficulty
        complexity_factor = 1.0 + (memory.complexity_score / 10.0)
        difficulty_factor = 1.0 + (memory.difficulty / 15.0)

        # Adjust for cognitive capacity
        capacity_factor = 2.0 - cognitive_capacity  # Lower capacity = more time

        estimated_duration = (
            base_time * complexity_factor * difficulty_factor * capacity_factor
        )

        return max(0.5, min(15.0, estimated_duration))  # Cap between 30s and 15min


class ExamPressureSimulator:
    """Models the impact of exam pressure on learning and recall"""

    def __init__(self):
        self.stress_onset_threshold = 14  # Days before exam when stress begins
        self.panic_threshold = 3  # Days before exam when panic sets in
        self.cramming_effectiveness = CRAMMING_EFFECTIVENESS_DECAY

    def calculate_exam_stress_level(
        self, days_to_exam: int, preparation_level: float
    ) -> float:
        """Calculate stress level based on exam proximity and preparation"""
        if days_to_exam > self.stress_onset_threshold:
            return 0.1  # Minimal background stress

        # Stress increases exponentially as exam approaches
        time_pressure = 1.0 - (days_to_exam / self.stress_onset_threshold)

        # Poor preparation amplifies stress
        preparation_anxiety = max(0.0, 0.8 - preparation_level)

        # Panic phase
        if days_to_exam <= self.panic_threshold:
            panic_multiplier = 2.0 - (days_to_exam / self.panic_threshold)
            time_pressure *= panic_multiplier

        total_stress = min(1.0, time_pressure + preparation_anxiety)
        return total_stress

    def calculate_performance_degradation(
        self, stress_level: float, cramming_ratio: float
    ) -> float:
        """Calculate how stress and cramming affect performance"""
        # Stress impairs recall (Yerkes-Dodson law approximation)
        if stress_level < 0.3:
            stress_factor = 1.0 + stress_level * 0.1  # Slight improvement
        elif stress_level < 0.7:
            stress_factor = 1.0 - (stress_level - 0.3) * 0.2  # Linear decline
        else:
            stress_factor = 0.92 - (stress_level - 0.7) * 0.3  # Steep decline

        # Cramming penalty (spaced repetition is superior)
        cramming_penalty = cramming_ratio * (1.0 - self.cramming_effectiveness)

        total_degradation = stress_factor * (1.0 - cramming_penalty)
        return max(0.3, min(1.2, total_degradation))

    def adjust_retrievability_for_exam_context(
        self,
        base_retrievability: float,
        days_to_exam: int,
        cramming_ratio: float,
        preparation_level: float,
    ) -> float:
        """Adjust retrievability predictions for exam context"""
        if days_to_exam > 30:
            return base_retrievability  # No exam pressure yet

        stress_level = self.calculate_exam_stress_level(days_to_exam, preparation_level)
        performance_factor = self.calculate_performance_degradation(
            stress_level, cramming_ratio
        )

        exam_adjusted = base_retrievability * performance_factor

        # Add some randomness for test anxiety
        anxiety_variance = random.uniform(-0.05, 0.05) * stress_level

        return max(0.05, min(0.98, exam_adjusted + anxiety_variance))


class PerformanceValidator:
    """Validates predictions against reality and provides confidence intervals"""

    def __init__(self):
        self.prediction_history = []
        self.calibration_data = defaultdict(list)
        self.confidence_threshold = 0.8

    def validate_retrievability_prediction(
        self, predicted: float, actual: float, memory: TopicMemory
    ) -> Dict:
        """Compare prediction vs actual performance"""
        error = abs(predicted - actual)
        relative_error = error / max(0.1, actual)  # Avoid division by zero

        # Store for calibration
        self.prediction_history.append(
            {
                "predicted": predicted,
                "actual": actual,
                "error": error,
                "relative_error": relative_error,
                "state": memory.state,
                "difficulty": memory.difficulty,
                "timestamp": datetime.now(),
            }
        )

        # Calculate prediction quality
        quality = (
            "excellent"
            if error < 0.05
            else "good" if error < 0.15 else "fair" if error < 0.25 else "poor"
        )

        return {
            "error": error,
            "relative_error": relative_error,
            "quality": quality,
            "confidence": self.calculate_prediction_confidence(memory),
        }

    def calculate_prediction_confidence(self, memory: TopicMemory) -> float:
        """Calculate confidence in retrievability prediction"""
        base_confidence = 0.5

        # More reviews = more confidence
        review_confidence = min(0.3, memory.review_count * 0.01)

        # Stable patterns = more confidence
        if len(memory.recent_ratings) >= 5:
            rating_variance = statistics.variance(
                [float(r) for r in memory.recent_ratings]
            )
            stability_confidence = max(0.0, 0.2 - rating_variance * 0.05)
        else:
            stability_confidence = 0.0

        # State-based confidence
        state_confidence = {
            TopicState.NEW: 0.1,
            TopicState.LEARNING: 0.3,
            TopicState.YOUNG: 0.6,
            TopicState.REVIEW: 0.8,
            TopicState.MATURE: 0.9,
            TopicState.RELEARNING: 0.4,
        }.get(memory.state, 0.5)

        total_confidence = min(
            0.95,
            base_confidence
            + review_confidence
            + stability_confidence
            + state_confidence * 0.3,
        )

        return total_confidence

    def get_confidence_interval(
        self, predicted_value: float, confidence_level: float = 0.8
    ) -> Tuple[float, float]:
        """Calculate confidence interval around prediction"""
        if len(self.prediction_history) < 10:
            # Not enough data, use conservative estimates
            margin = predicted_value * 0.3
        else:
            # Use historical error distribution
            recent_errors = [p["error"] for p in self.prediction_history[-50:]]
            margin = statistics.quantile(recent_errors, 1 - (1 - confidence_level) / 2)

        lower_bound = max(0.0, predicted_value - margin)
        upper_bound = min(1.0, predicted_value + margin)

        return (lower_bound, upper_bound)


class StudySessionOptimizer:
    """Optimizes study sessions for maximum effectiveness"""

    def __init__(self):
        self.cognitive_manager = CognitiveLoadManager()
        self.exam_simulator = ExamPressureSimulator()
        self.optimal_session_duration = 45  # minutes
        self.max_topics_per_session = 15

    def plan_study_session(
        self,
        available_topics: List[TopicMemory],
        cognitive_state: CognitiveState,
        time_available: float,
    ) -> Dict:
        """Plan an optimal study session"""
        current_capacity = self.cognitive_manager.calculate_current_capacity(
            cognitive_state, []
        )

        # Filter topics that need review
        due_topics = [
            t
            for t in available_topics
            if t.due_date <= datetime.now() or self._is_exam_urgent(t)
        ]

        # Sort by priority (considering cognitive load)
        prioritized_topics = self._prioritize_topics(due_topics, cognitive_state)

        # Select topics for session
        selected_topics = self._select_session_topics(
            prioritized_topics, current_capacity, time_available
        )

        # Estimate session metrics
        estimated_duration = sum(
            self.cognitive_manager.estimate_review_duration(topic, current_capacity)
            for topic in selected_topics
        )

        expected_fatigue = (
            len(selected_topics) * self.cognitive_manager.fatigue_accumulation_rate
        )

        return {
            "selected_topics": selected_topics,
            "estimated_duration_minutes": estimated_duration,
            "expected_fatigue_increase": expected_fatigue,
            "cognitive_capacity_required": current_capacity,
            "session_efficiency_score": self._calculate_session_efficiency(
                selected_topics
            ),
            "recommendations": self._generate_session_recommendations(
                selected_topics, cognitive_state
            ),
        }

    def _prioritize_topics(
        self, topics: List[TopicMemory], cognitive_state: CognitiveState
    ) -> List[TopicMemory]:
        """Priority score combining urgency, difficulty, and cognitive fit"""
        scored_topics = []

        for topic in topics:
            # Base urgency (overdue-ness)
            days_overdue = max(0, (datetime.now() - topic.due_date).days)
            urgency_score = min(1.0, days_overdue / 7.0)  # Cap at 1 week overdue

            # Exam urgency
            exam_urgency = 0.0
            if topic.exam_dates:
                nearest_exam_days = min(
                    (exam - date.today()).days for exam in topic.exam_dates
                )
                if nearest_exam_days <= 30:
                    exam_urgency = 1.0 - (nearest_exam_days / 30.0)

            # Forgetting risk (retrievability decline)
            current_retrievability = CoreFSRSEngine().calculate_retrievability(
                topic.stability,
                (
                    (datetime.now() - topic.last_review_date).days
                    if topic.last_review_date
                    else 1
                ),
            )
            forgetting_risk = 1.0 - current_retrievability

            # Cognitive fit (easier topics when tired)
            cognitive_fit = 1.0
            if cognitive_state.current_capacity < 0.7:  # Tired
                cognitive_fit = 1.0 - (topic.difficulty / 10.0)

            # Combined priority
            total_priority = (
                urgency_score * 0.3
                + exam_urgency * 0.4
                + forgetting_risk * 0.2
                + cognitive_fit * 0.1
            )

            scored_topics.append((topic, total_priority))

        # Sort by priority (highest first)
        scored_topics.sort(key=lambda x: x[1], reverse=True)
        return [topic for topic, score in scored_topics]

    def _select_session_topics(
        self,
        prioritized_topics: List[TopicMemory],
        capacity: float,
        time_available: float,
    ) -> List[TopicMemory]:
        """Select topics that fit time and cognitive constraints"""
        selected = []
        total_time = 0.0
        complexity_sum = 0.0

        for topic in prioritized_topics:
            estimated_time = self.cognitive_manager.estimate_review_duration(
                topic, capacity
            )

            # Check time constraint
            if total_time + estimated_time > time_available:
                break

            # Check cognitive load constraint
            if len(selected) >= self.max_topics_per_session:
                break

            # Check complexity overload
            if (
                complexity_sum + topic.complexity_score > capacity * 50
            ):  # Arbitrary threshold
                break

            selected.append(topic)
            total_time += estimated_time
            complexity_sum += topic.complexity_score

        return selected

    def _is_exam_urgent(self, topic: TopicMemory) -> bool:
        """Check if topic needs urgent review due to upcoming exam"""
        if not topic.exam_dates:
            return False

        nearest_exam_days = min((exam - date.today()).days for exam in topic.exam_dates)
        return nearest_exam_days <= 7  # Urgent if exam within a week

    def _calculate_session_efficiency(self, topics: List[TopicMemory]) -> float:
        """Calculate expected learning efficiency of the session"""
        if not topics:
            return 0.0

        # Factors affecting efficiency
        state_distribution = defaultdict(int)
        for topic in topics:
            state_distribution[topic.state] += 1

        # Prefer mix of states for variety
        variety_score = len(state_distribution) / len(TopicState)

        # Prefer appropriate difficulty progression
        difficulties = sorted([t.difficulty for t in topics])
        progression_score = 1.0 - (max(difficulties) - min(difficulties)) / 10.0

        # Combined efficiency
        efficiency = variety_score * 0.4 + progression_score * 0.6
        return max(0.1, min(1.0, efficiency))

    def _generate_session_recommendations(
        self, topics: List[TopicMemory], cognitive_state: CognitiveState
    ) -> List[str]:
        """Generate actionable recommendations for the study session"""
        recommendations = []

        if cognitive_state.current_capacity < 0.5:
            recommendations.append(
                "Take a 15-minute break before starting - you seem fatigued"
            )

        if len(topics) > 10:
            recommendations.append("Consider splitting this into two shorter sessions")

        difficult_topics = [t for t in topics if t.difficulty > 7.0]
        if len(difficult_topics) > 3:
            recommendations.append("Start with easier topics to build momentum")

        new_topics = [t for t in topics if t.state == TopicState.NEW]
        if len(new_topics) > 5:
            recommendations.append("Focus on fewer new topics for better retention")

        if cognitive_state.consecutive_study_days > 6:
            recommendations.append("Consider taking a rest day to prevent burnout")

        return recommendations


class AdaptiveParameterLearner:
    """Learns and adjusts algorithm parameters based on user performance"""

    def __init__(self, initial_parameters: List[float] = None):
        self.parameters = (initial_parameters or DEFAULT_PARAMETERS).copy()
        self.performance_history = []
        self.parameter_adjustments = []
        self.learning_rate = 0.001  # Conservative learning
        self.minimum_data_points = 50  # Before making adjustments

    def record_performance_data(
        self,
        predicted_retrievability: float,
        actual_performance: Rating,
        topic_memory: TopicMemory,
        review_context: Dict,
    ):
        """Record prediction vs actual performance for learning"""
        # Convert rating to retention estimate
        actual_retention = self._rating_to_retention_estimate(
            actual_performance, review_context.get("response_time", 5.0)
        )

        data_point = {
            "timestamp": datetime.now(),
            "predicted": predicted_retrievability,
            "actual": actual_retention,
            "error": abs(predicted_retrievability - actual_retention),
            "squared_error": (predicted_retrievability - actual_retention) ** 2,
            "topic_state": topic_memory.state,
            "topic_difficulty": topic_memory.difficulty,
            "topic_stability": topic_memory.stability,
            "days_since_review": review_context.get("days_since_review", 0),
            "context": review_context,
        }

        self.performance_history.append(data_point)

        # Trim old data (keep last 1000 points)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def should_update_parameters(self) -> bool:
        """Determine if we have enough data for parameter updates"""
        if len(self.performance_history) < self.minimum_data_points:
            return False

        # Check if prediction accuracy is poor
        recent_errors = [p["error"] for p in self.performance_history[-20:]]
        mean_error = statistics.mean(recent_errors)

        return mean_error > 0.15  # Update if average error > 15%

    def update_parameters(self):
        """Update parameters using gradient descent on prediction error"""
        if not self.should_update_parameters():
            return

        # Use recent performance for updates
        recent_data = self.performance_history[-100:]

        # Calculate gradients for key parameters
        gradients = self._calculate_parameter_gradients(recent_data)

        # Update parameters conservatively
        for i, gradient in enumerate(gradients):
            if abs(gradient) > 0.001:  # Only update if gradient is significant
                adjustment = -self.learning_rate * gradient
                self.parameters[i] += adjustment

                # Keep parameters in reasonable bounds
                self.parameters[i] = self._clamp_parameter(i, self.parameters[i])

        # Record the adjustment
        self.parameter_adjustments.append(
            {
                "timestamp": datetime.now(),
                "mean_error_before": statistics.mean([p["error"] for p in recent_data]),
                "adjustments": gradients,
                "parameter_snapshot": self.parameters.copy(),
            }
        )

    def _rating_to_retention_estimate(
        self, rating: Rating, response_time: float
    ) -> float:
        """Convert rating and response time to retention estimate"""
        base_retention = {
            Rating.BLACKOUT: 0.0,
            Rating.AGAIN: 0.3,
            Rating.HARD: 0.6,
            Rating.GOOD: 0.8,
            Rating.EASY: 0.95,
        }.get(rating, 0.5)

        # Adjust based on response time (faster = better retention)
        time_adjustment = 0.0
        if response_time < 2.0:  # Very fast
            time_adjustment = 0.1
        elif response_time > 10.0:  # Very slow
            time_adjustment = -0.1

        return max(0.0, min(1.0, base_retention + time_adjustment))

    def _calculate_parameter_gradients(self, data_points: List[Dict]) -> List[float]:
        """Calculate gradients for parameter updates (simplified)"""
        gradients = [0.0] * len(self.parameters)

        # Focus on stability-related parameters (most impactful)
        stability_params = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14]

        for param_idx in stability_params:
            if param_idx >= len(self.parameters):
                continue

            gradient = self._calculate_single_gradient(param_idx, data_points)
            gradients[param_idx] = gradient

        return gradients

    def _calculate_single_gradient(
        self, param_idx: int, data_points: List[Dict]
    ) -> float:
        """Calculate gradient for a single parameter"""
        total_gradient = 0.0
        epsilon = 0.001  # Small perturbation for numerical gradient

        for data_point in data_points[-20:]:  # Use subset for efficiency
            # Calculate prediction with current parameter
            original_param = self.parameters[param_idx]

            # Perturb parameter slightly
            self.parameters[param_idx] += epsilon
            prediction_plus = self._predict_with_current_params(data_point)

            self.parameters[param_idx] -= 2 * epsilon
            prediction_minus = self._predict_with_current_params(data_point)

            # Restore original parameter
            self.parameters[param_idx] = original_param

            # Calculate numerical gradient
            gradient = (prediction_plus - prediction_minus) / (2 * epsilon)
            error = data_point["predicted"] - data_point["actual"]

            total_gradient += gradient * error

        return total_gradient / len(data_points[-20:])

    def _predict_with_current_params(self, data_point: Dict) -> float:
        """Make prediction using current parameters"""
        fsrs = CoreFSRSEngine(self.parameters)
        return fsrs.calculate_retrievability(
            data_point["topic_stability"], data_point["days_since_review"]
        )

    def _clamp_parameter(self, param_idx: int, value: float) -> float:
        """Keep parameters within reasonable bounds"""
        bounds = {
            # Initial stability parameters
            0: (0.1, 10.0),
            1: (0.5, 5.0),
            2: (1.0, 10.0),
            3: (5.0, 20.0),
            # Other parameters with conservative bounds
        }

        if param_idx in bounds:
            min_val, max_val = bounds[param_idx]
            return max(min_val, min(value, max_val))

        # Default bounds for other parameters
        return max(-10.0, min(10.0, value))


class ComprehensiveTopicScheduler:
    """Main scheduler class that coordinates all components"""

    def __init__(self, parameters: List[float] = None):
        self.fsrs_engine = CoreFSRSEngine(parameters)
        self.state_manager = MemoryStateManager()
        self.cognitive_manager = CognitiveLoadManager()
        self.exam_simulator = ExamPressureSimulator()
        self.performance_validator = PerformanceValidator()
        self.session_optimizer = StudySessionOptimizer()
        self.adaptive_learner = AdaptiveParameterLearner(parameters)

        # Learning steps (conservative progression)
        self.learning_steps = [
            timedelta(hours=4),
            timedelta(hours=12),  # overnight break
            timedelta(days=2),  # 2 days
            timedelta(days=4),  # 3 days
        ]

        self.relearning_steps = [
            timedelta(hours=6),
            timedelta(days=1),
            timedelta(days=3),  # 3 days
        ]

        self.complexity_modifiers = {
            1.0: 0.7,  # Very simple topics can be reviewed sooner
            2.0: 0.8,
            3.0: 0.9,
            4.0: 1.0,  # Average complexity
            5.0: 1.1,
            6.0: 1.2,
            7.0: 1.4,  # Complex topics need more time
            8.0: 1.6,
            9.0: 1.8,
            10.0: 2.0,  # Very complex topics need double time
        }

    def process_review(
        self,
        memory: TopicMemory,
        rating: Rating,
        response_time_seconds: float = None,
        study_context: Dict = None,
    ) -> Dict:
        """Process a review and update all topic parameters"""
        review_datetime = datetime.now()

        # Calculate days since last review
        days_since_last = 0.0
        if memory.last_review_date:
            days_since_last = (
                review_datetime - memory.last_review_date
            ).total_seconds() / 86400

        # Get current retrievability before update
        pre_review_retrievability = self.fsrs_engine.calculate_retrievability(
            memory.stability, days_since_last
        )

        # Update FSRS parameters
        if rating <= Rating.AGAIN:
            # Failed recall - update for forgetting
            new_stability = self.fsrs_engine.calculate_next_stability_failure(
                memory.stability, memory.difficulty, pre_review_retrievability
            )
            memory.lapses += 1
            memory.streak = 0
        else:
            # Successful recall
            new_stability = self.fsrs_engine.calculate_next_stability_success(
                memory.stability, memory.difficulty, pre_review_retrievability, rating
            )
            memory.streak += 1

        new_difficulty = self.fsrs_engine.calculate_next_difficulty(
            memory.difficulty, rating
        )

        # Determine new state
        new_state = self.state_manager.determine_next_state(
            memory, rating, days_since_last
        )

        # Calculate next due date
        next_due = self._calculate_next_due_date(
            memory, new_state, new_stability, rating, review_datetime
        )

        # Update memory object
        memory.stability = new_stability
        memory.difficulty = new_difficulty
        memory.state = new_state
        memory.last_review_date = review_datetime
        memory.due_date = next_due
        memory.review_count += 1

        # Update performance tracking
        memory.recent_ratings = (memory.recent_ratings + [rating])[-10:]
        if response_time_seconds:
            memory.response_times = (memory.response_times + [response_time_seconds])[
                -20:
            ]

        # Calculate retention estimate for this review
        retention_estimate = self.adaptive_learner._rating_to_retention_estimate(
            rating, response_time_seconds or 5.0
        )

        # Update average retention (exponential moving average)
        alpha = 0.2  # Learning rate for average
        memory.average_retention = (
            alpha * retention_estimate + (1 - alpha) * memory.average_retention
        )

        # Record performance for adaptive learning
        review_context = {
            "days_since_review": days_since_last,
            "response_time": response_time_seconds or 5.0,
            "study_context": study_context or {},
        }

        self.adaptive_learner.record_performance_data(
            pre_review_retrievability, rating, memory, review_context
        )

        # Validate prediction accuracy
        validation_result = (
            self.performance_validator.validate_retrievability_prediction(
                pre_review_retrievability, retention_estimate, memory
            )
        )

        # Get confidence interval for next prediction
        post_review_retrievability = self.fsrs_engine.calculate_retrievability(
            memory.stability, 0  # Just reviewed
        )
        confidence_interval = self.performance_validator.get_confidence_interval(
            post_review_retrievability
        )

        return {
            "updated_memory": memory,
            "pre_review_retrievability": pre_review_retrievability,
            "post_review_retrievability": post_review_retrievability,
            "confidence_interval": confidence_interval,
            "validation_result": validation_result,
            "next_review_date": next_due,
            "state_changed": new_state != memory.state,
            "recommendation": self._generate_review_recommendation(
                memory, rating, validation_result
            ),
        }

    def calculate_realistic_topic_strength(
        self,
        memory: TopicMemory,
        current_datetime: datetime = None,
        exam_context: Dict = None,
    ) -> Dict:
        """Calculate comprehensive, realistic topic strength assessment"""
        if current_datetime is None:
            current_datetime = datetime.now()

        # Base retrievability from FSRS
        days_since_review = 0.0
        if memory.last_review_date:
            days_since_review = (
                current_datetime - memory.last_review_date
            ).total_seconds() / 86400

        base_retrievability = self.fsrs_engine.calculate_retrievability(
            memory.stability, days_since_review
        )

        if memory.review_count <= 3 or len(memory.recent_ratings) <= 2:
            if memory.recent_ratings:
                latest_rating = memory.recent_ratings[-1]
                # Map ratings to more realistic retrievability caps
                rating_caps = {
                    Rating.BLACKOUT: 0.15,
                    Rating.AGAIN: 0.35,
                    Rating.HARD: 0.55,
                    Rating.GOOD: 0.75,
                    Rating.EASY: 0.85,
                }
                max_retrievability = rating_caps.get(latest_rating, 0.5)
                base_retrievability = min(base_retrievability, max_retrievability)
            else:
                base_retrievability = 0.3

        # Apply exam pressure adjustments if applicable
        exam_adjusted_retrievability = base_retrievability
        exam_stress_level = 0.0

        if exam_context and memory.exam_dates:
            nearest_exam_days = min(
                (exam_date - current_datetime.date()).days
                for exam_date in memory.exam_dates
                if exam_date >= current_datetime.date()
            )

            if nearest_exam_days <= 30:
                preparation_level = exam_context.get("overall_preparation", 0.7)
                cramming_ratio = memory.cramming_sessions / max(1, memory.review_count)

                exam_adjusted_retrievability = (
                    self.exam_simulator.adjust_retrievability_for_exam_context(
                        base_retrievability,
                        nearest_exam_days,
                        cramming_ratio,
                        preparation_level,
                    )
                )

                exam_stress_level = self.exam_simulator.calculate_exam_stress_level(
                    nearest_exam_days, preparation_level
                )

        # Get confidence interval
        confidence_interval = self.performance_validator.get_confidence_interval(
            exam_adjusted_retrievability
        )

        # Calculate maturity score based on stability and review history
        maturity_score = self._calculate_maturity_score(memory)

        # Determine readiness category (much more conservative)
        if exam_adjusted_retrievability >= 0.90 and maturity_score >= 0.8:
            readiness = "excellent"
            readiness_score = 0.95
        elif exam_adjusted_retrievability >= 0.80 and maturity_score >= 0.6:
            readiness = "good"
            readiness_score = 0.80
        elif exam_adjusted_retrievability >= 0.70 and maturity_score >= 0.4:
            readiness = "fair"
            readiness_score = 0.65
        elif exam_adjusted_retrievability >= 0.60:
            readiness = "poor"
            readiness_score = 0.45
        else:
            readiness = "critical"
            readiness_score = 0.25

        return {
            "base_retrievability": round(base_retrievability, 3),
            "exam_adjusted_retrievability": round(exam_adjusted_retrievability, 3),
            "confidence_interval": {
                "lower": round(confidence_interval[0], 3),
                "upper": round(confidence_interval[1], 3),
            },
            "readiness_category": readiness,
            "readiness_score": readiness_score,
            "maturity_score": round(maturity_score, 3),
            "exam_stress_level": round(exam_stress_level, 3),
            "stability_days": round(memory.stability, 1),
            "difficulty_score": round(memory.difficulty, 1),
            "review_history_strength": self._calculate_review_history_strength(memory),
            "prediction_confidence": self.performance_validator.calculate_prediction_confidence(
                memory
            ),
            "days_since_last_review": (
                round(days_since_review, 1) if days_since_review > 0 else 0
            ),
            "state_description": self._get_state_description(memory.state),
            "improvement_potential": self._calculate_improvement_potential(memory),
        }

    def _calculate_next_due_date(
        self,
        memory: TopicMemory,
        new_state: TopicState,
        stability: float,
        rating: Rating,
        review_datetime: datetime,
    ) -> datetime:
        """Calculate when topic should next be reviewed"""
        
        # Get complexity modifier (from topic complexity score)
        complexity_factor = self._get_complexity_modifier(memory.complexity_score)
        
        if new_state == TopicState.NEW:
            # First review after 4 hours (adjusted for complexity)
            base_interval = timedelta(hours=4)
            return review_datetime + timedelta(seconds=base_interval.total_seconds() * complexity_factor)
        
        elif new_state == TopicState.LEARNING:
            # FIXED: Check for early graduation with EASY rating
            if rating == Rating.EASY and memory.review_count >= 1:
                # Graduate immediately to stability-based interval
                base_interval_days = max(2.0, stability * 0.5)  # At least 2 days, conservative
                adjusted_interval = timedelta(days=base_interval_days * complexity_factor)
                return review_datetime + adjusted_interval
            
            # Use learning steps with complexity adjustment
            step_index = min(memory.review_count - 1, len(self.learning_steps) - 1)
            if rating <= Rating.AGAIN:
                step_index = 0  # Reset to first step (but still 4 hours minimum)
            
            base_interval = self.learning_steps[step_index]
            adjusted_interval = timedelta(seconds=base_interval.total_seconds() * complexity_factor)
            
            # Additional penalty for repeated failures
            if rating <= Rating.AGAIN and memory.lapses > 1:
                failure_penalty = min(2.0, 1.0 + (memory.lapses * 0.3))
                adjusted_interval = timedelta(seconds=adjusted_interval.total_seconds() * failure_penalty)
            
            return review_datetime + adjusted_interval
        
        elif new_state == TopicState.RELEARNING:
            # Use relearning steps with even more generous intervals
            step_index = min(memory.lapses - 1, len(self.relearning_steps) - 1)
            if rating <= Rating.AGAIN:
                step_index = 0
            
            base_interval = self.relearning_steps[step_index]
            
            # Relearning gets extra time based on how many times forgotten
            relearning_penalty = 1.0 + (memory.lapses * 0.2)  # 20% more time per lapse
            
            adjusted_interval = timedelta(
                seconds=base_interval.total_seconds() * complexity_factor * relearning_penalty
            )
            
            return review_datetime + adjusted_interval
        
        else:
            # YOUNG, REVIEW, MATURE states use stability-based intervals
            base_interval_days = stability
            
            # Apply complexity factor to stability-based intervals too
            base_interval_days *= complexity_factor
            
            # Apply state modifier
            state_modifier = self.state_manager.calculate_state_interval_modifier(new_state)
            interval_days = base_interval_days * state_modifier
            
            # Rating-based adjustment for stability intervals
            rating_modifiers = {
                Rating.AGAIN: 0.5,     # Cut interval in half if struggled
                Rating.HARD: 0.7,      # Reduce interval for difficult recall
                Rating.GOOD: 1.0,      # Normal interval
                Rating.EASY: 1.3       # Slightly longer for easy recall
            }
            interval_days *= rating_modifiers.get(rating, 1.0)
            
            # Apply exam pressure compression if needed
            if memory.exam_dates:
                nearest_exam_days = min(
                    (exam_date - review_datetime.date()).days 
                    for exam_date in memory.exam_dates 
                    if exam_date >= review_datetime.date()
                )
                
                if nearest_exam_days <= 30:
                    compression_factor = max(0.4, nearest_exam_days / 30.0)  # More aggressive compression
                    interval_days *= compression_factor
            
            # Clamp to reasonable bounds (minimum 6 hours for topics, not minutes)
            interval_days = max(0.25, min(interval_days, 365.0))  # Min 6 hours, max 1 year
            
            # Add some randomization to avoid clustering
            fuzz_factor = random.uniform(0.95, 1.05)  # Smaller fuzz for topics
            interval_days *= fuzz_factor
            
            return review_datetime + timedelta(days=interval_days)

    def _get_complexity_modifier(self, complexity_score: float) -> float:
        """Get interval modifier based on topic complexity"""
        # Linear interpolation between complexity levels
        if complexity_score <= 1.0:
            return 0.7
        elif complexity_score >= 10.0:
            return 2.0
        else:
            # Linear interpolation
            lower_bound = int(complexity_score)
            upper_bound = lower_bound + 1
            
            if lower_bound in self.complexity_modifiers and upper_bound in self.complexity_modifiers:
                lower_modifier = self.complexity_modifiers[lower_bound]
                upper_modifier = self.complexity_modifiers[upper_bound]
                
                # Interpolate
                weight = complexity_score - lower_bound
                return lower_modifier + (upper_modifier - lower_modifier) * weight
            
            # Fallback
            return 1.0 + (complexity_score - 5.0) * 0.1

    def _calculate_maturity_score(self, memory: TopicMemory) -> float:
        """Calculate how mature/consolidated the memory is"""
        factors = []

        # Review count factor (more reviews = more mature)
        review_factor = min(1.0, memory.review_count / 20.0)
        factors.append(review_factor * 0.3)

        # Streak factor (consistent success)
        streak_factor = min(1.0, memory.streak / 8.0)
        factors.append(streak_factor * 0.2)

        # Stability factor (longer intervals = more stable)
        stability_factor = min(1.0, memory.stability / 90.0)  # 90 days = fully mature
        factors.append(stability_factor * 0.3)

        # Time factor (older memories are more consolidated)
        if memory.initial_learning_date:
            days_learning = (datetime.now() - memory.initial_learning_date).days
            time_factor = min(1.0, days_learning / 180.0)  # 6 months = mature
            factors.append(time_factor * 0.2)
        else:
            factors.append(0.0)

        return sum(factors)

    def _calculate_review_history_strength(self, memory: TopicMemory) -> float:
        """Assess the strength of review history pattern"""
        if not memory.recent_ratings:
            return 0.0

        # Average rating (higher is better)
        avg_rating = statistics.mean([float(r) for r in memory.recent_ratings])
        rating_strength = (avg_rating - 2.0) / 3.0  # Normalize to 0-1

        # Consistency (less variance is better)
        if len(memory.recent_ratings) > 1:
            rating_variance = statistics.variance(
                [float(r) for r in memory.recent_ratings]
            )
            consistency_strength = max(0.0, 1.0 - rating_variance / 2.0)
        else:
            consistency_strength = 0.5

        # Recent improvement trend
        if len(memory.recent_ratings) >= 5:
            recent_half = memory.recent_ratings[-3:]
            older_half = (
                memory.recent_ratings[-6:-3]
                if len(memory.recent_ratings) >= 6
                else memory.recent_ratings[:-3]
            )

            if older_half:
                recent_avg = statistics.mean([float(r) for r in recent_half])
                older_avg = statistics.mean([float(r) for r in older_half])
                trend_strength = max(
                    0.0, min(1.0, 0.5 + (recent_avg - older_avg) / 2.0)
                )
            else:
                trend_strength = 0.5
        else:
            trend_strength = 0.5

        # Combined strength
        return rating_strength * 0.5 + consistency_strength * 0.3 + trend_strength * 0.2

    def _calculate_improvement_potential(self, memory: TopicMemory) -> float:
        """Calculate how much the topic could potentially improve"""
        # Based on current difficulty and performance
        difficulty_potential = (
            memory.difficulty / 10.0
        )  # Higher difficulty = more room to improve

        # Based on recent performance
        performance_potential = 1.0 - memory.average_retention

        # Based on state (some states have more potential than others)
        state_potential = {
            TopicState.NEW: 1.0,
            TopicState.LEARNING: 0.8,
            TopicState.YOUNG: 0.6,
            TopicState.REVIEW: 0.4,
            TopicState.MATURE: 0.2,
            TopicState.RELEARNING: 0.7,
        }.get(memory.state, 0.5)

        # Combined potential
        return (
            difficulty_potential * 0.4
            + performance_potential * 0.4
            + state_potential * 0.2
        )

    def _get_state_description(self, state: TopicState) -> str:
        """Get human-readable state description"""
        descriptions = {
            TopicState.NEW: "Never studied - needs initial learning",
            TopicState.LEARNING: "Currently learning - building initial memory",
            TopicState.YOUNG: "Recently learned - memory consolidating",
            TopicState.REVIEW: "Under review - stable but needs reinforcement",
            TopicState.MATURE: "Well learned - long-term memory established",
            TopicState.RELEARNING: "Being relearned - recovering from forgetting",
        }
        return descriptions.get(state, "Unknown state")

    def _generate_review_recommendation(
        self, memory: TopicMemory, last_rating: Rating, validation_result: Dict
    ) -> str:
        """Generate actionable recommendation based on review performance"""
        recommendations = []

        if last_rating <= Rating.AGAIN:
            recommendations.append(
                "Focus on understanding fundamentals before proceeding"
            )
        elif last_rating == Rating.HARD:
            recommendations.append("Review related concepts to strengthen connections")
        elif last_rating >= Rating.GOOD:
            recommendations.append("Good progress! Continue with spaced reviews")

        if validation_result["quality"] == "poor":
            recommendations.append(
                "Performance was unpredictable - review study methods"
            )

        if memory.difficulty > 8.0:
            recommendations.append("Consider breaking this topic into smaller chunks")

        if memory.streak >= 5:
            recommendations.append(
                "Excellent consistency! Memory is consolidating well"
            )

        return (
            "; ".join(recommendations)
            if recommendations
            else "Continue regular review schedule"
        )


# Usage example and testing functions
def create_sample_topic_memory(name: str, subject: str = None) -> TopicMemory:
    """Create a sample topic memory for testing"""
    return TopicMemory(
        stability=1.0,
        difficulty=5.0,
        initial_learning_date=datetime.now() - timedelta(days=30),
        last_review_date=datetime.now() - timedelta(days=3),
        due_date=datetime.now() + timedelta(days=1),
        state=TopicState.REVIEW,
        review_count=10,
        lapses=1,
        streak=3,
        average_retention=0.75,
        response_times=[3.2, 4.1, 2.8, 5.2],
        recent_ratings=[
            Rating.GOOD,
            Rating.GOOD,
            Rating.HARD,
            Rating.GOOD,
            Rating.EASY,
        ],
        similar_topics=[],
        subject_category=subject,
        complexity_score=6.0,
        exam_dates=[date.today() + timedelta(days=14)],
        exam_weights=[1.0],
        cramming_sessions=2,
    )


def run_algorithm_validation_test():
    """Test the algorithm with sample data"""
    scheduler = ComprehensiveTopicScheduler()

    # Create test topic
    topic = create_sample_topic_memory("Photosynthesis", "Biology")

    # Process a review
    result = scheduler.process_review(topic, Rating.GOOD, response_time_seconds=4.5)

    # Calculate topic strength
    strength = scheduler.calculate_realistic_topic_strength(
        topic, exam_context={"overall_preparation": 0.6}
    )

    print("=== Algorithm Test Results ===")
    print(f"Topic: Photosynthesis")
    print(f"Pre-review retrievability: {result['pre_review_retrievability']:.3f}")
    print(f"Post-review retrievability: {result['post_review_retrievability']:.3f}")
    print(
        f"Exam-adjusted retrievability: {strength['exam_adjusted_retrievability']:.3f}"
    )
    print(
        f"Readiness: {strength['readiness_category']} ({strength['readiness_score']:.2f})"
    )
    print(f"Maturity score: {strength['maturity_score']:.3f}")
    print(f"Next review: {result['next_review_date'].strftime('%Y-%m-%d %H:%M')}")
    print(f"Recommendation: {result['recommendation']}")
    print(
        f"Confidence interval: {strength['confidence_interval']['lower']:.3f} - {strength['confidence_interval']['upper']:.3f}"
    )


if __name__ == "__main__":
    run_algorithm_validation_test()
