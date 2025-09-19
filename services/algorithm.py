import math
from dataclasses import dataclass
from datetime import timedelta
from random import random
from constants import *

from utils.datetime_utils import ensure_timezone_aware, now_ist
from models import Topic,TopicReviewLog



class TopicScheduler:
    """
    FSRS Scheduler adapted for Topics instead of Cards
    Maintains full FSRS algorithm rigor while adding exam-based prioritization
    """
    
    def __init__(self, parameters=None, desired_retention=0.9, 
                 learning_steps=None, relearning_steps=None, 
                 maximum_interval=36500, enable_fuzzing=True):
        
        self.parameters = parameters or DEFAULT_PARAMETERS
        self.desired_retention = desired_retention
        self.learning_steps = learning_steps or [
            timedelta(minutes=1), timedelta(minutes=10)
        ]
        self.relearning_steps = relearning_steps or [timedelta(minutes=10)]
        self.maximum_interval = maximum_interval
        self.enable_fuzzing = enable_fuzzing
        
        # Pre-calculate FSRS constants
        self._DECAY = -self.parameters[20]
        self._FACTOR = 0.9 ** (1 / self._DECAY) - 1
    
    def _clamp_difficulty(self, difficulty):
        """Clamp difficulty to valid range [1, 10]"""
        return max(MIN_DIFFICULTY, min(difficulty, MAX_DIFFICULTY))
    
    def _clamp_stability(self, stability):
        """Clamp stability to valid range [STABILITY_MIN, ∞)"""
        return max(stability, STABILITY_MIN)
    
    def _initial_stability(self, rating):
        """Calculate initial stability: S₀(G) = w[G-1], S₀ = max{S₀, 0.1}"""
        initial_stability = self.parameters[rating - 1]
        return self._clamp_stability(initial_stability)
    
    def _initial_difficulty(self, rating, clamp=True):
        """Calculate initial difficulty: D₀(G) = w₄ - e^((G-1) x w₅) + 1"""
        initial_difficulty = (
            self.parameters[4] - 
            (math.e ** (self.parameters[5] * (rating - 1))) + 1
        )
        
        if clamp:
            initial_difficulty = self._clamp_difficulty(initial_difficulty)
        
        return initial_difficulty
    
    def _next_interval(self, stability):
        """Calculate next interval: I(r,s) = (r^(1/DECAY) - 1) / FACTOR x s"""
        next_interval = (stability / self._FACTOR) * (
            (self.desired_retention ** (1 / self._DECAY)) - 1
        )
        
        next_interval = round(float(next_interval))
        next_interval = max(next_interval, 1)  # At least 1 day
        next_interval = min(next_interval, self.maximum_interval)
        
        return next_interval
    
    def _short_term_stability(self, stability, rating):
        """Calculate short-term stability: S'ₛ(S,G) = S x e^(w₁₇ x (G-3+w₁₈))"""
        stability_increase = (
            math.e ** (self.parameters[17] * (rating - 3 + self.parameters[18]))
        ) * (stability ** -self.parameters[19])
        
        if rating in (Rating.Good, Rating.Easy):
            stability_increase = max(stability_increase, 1.0)
        
        short_term_stability = stability * stability_increase
        return self._clamp_stability(short_term_stability)
    
    def _next_difficulty(self, difficulty, rating):
        """Calculate next difficulty with mean reversion and linear damping"""
        def linear_damping(delta_difficulty, difficulty):
            return (10.0 - difficulty) * delta_difficulty / 9.0
        
        def mean_reversion(arg_1, arg_2):
            return self.parameters[7] * arg_1 + (1 - self.parameters[7]) * arg_2
        
        # Mean reversion target (initial difficulty for Easy rating)
        arg_1 = self._initial_difficulty(Rating.Easy, clamp=False)
        
        # Calculate difficulty change
        delta_difficulty = -(self.parameters[6] * (rating - 3))
        arg_2 = difficulty + linear_damping(delta_difficulty, difficulty)
        
        next_difficulty = mean_reversion(arg_1, arg_2)
        return self._clamp_difficulty(next_difficulty)
    
    def _next_recall_stability(self, difficulty, stability, retrievability, rating):
        """Calculate stability after successful recall"""
        hard_penalty = self.parameters[15] if rating == Rating.Hard else 1
        easy_bonus = self.parameters[16] if rating == Rating.Easy else 1
        
        next_stability = stability * (
            1 + 
            (math.e ** self.parameters[8]) *
            (11 - difficulty) *
            (stability ** -self.parameters[9]) *
            ((math.e ** ((1 - retrievability) * self.parameters[10])) - 1) *
            hard_penalty *
            easy_bonus
        )
        
        return self._clamp_stability(next_stability)
    
    def _next_forget_stability(self, difficulty, stability, retrievability):
        """Calculate stability after forgetting"""
        forget_stability_long_term = (
            self.parameters[11] *
            (difficulty ** -self.parameters[12]) *
            (((stability + 1) ** self.parameters[13]) - 1) *
            (math.e ** ((1 - retrievability) * self.parameters[14]))
        )
        
        # Short-term constraint
        forget_stability_short_term = stability / (
            math.e ** (self.parameters[17] * self.parameters[18])
        )
        
        next_stability = min(forget_stability_long_term, forget_stability_short_term)
        return self._clamp_stability(next_stability)
    
    def _get_fuzzed_interval(self, interval_days):
        """Apply random fuzz to interval to avoid review clustering"""
        if not self.enable_fuzzing or interval_days < 2.5:
            return interval_days
        
        def get_fuzz_range(interval_days):
            delta = 1.0
            for fuzz_range in FUZZ_RANGES:
                delta += fuzz_range["factor"] * max(
                    min(interval_days, fuzz_range["end"]) - fuzz_range["start"], 0.0
                )
            
            min_ivl = int(round(interval_days - delta))
            max_ivl = int(round(interval_days + delta))
            
            min_ivl = max(2, min_ivl)
            max_ivl = min(max_ivl, self.maximum_interval)
            min_ivl = min(min_ivl, max_ivl)
            
            return min_ivl, max_ivl
        
        min_ivl, max_ivl = get_fuzz_range(interval_days)
        fuzzed_interval = (random() * (max_ivl - min_ivl + 1)) + min_ivl
        return min(round(fuzzed_interval), self.maximum_interval)
    
    def get_topic_retrievability(self, topic, current_datetime=None):
        """Calculate topic's current retrievability: R(t,S) = (1 + FACTOR × t/(9×S))^DECAY"""
        if topic.last_review is None or topic.stability is None:
            return 0.0
        
        if current_datetime is None:
            current_datetime = now_ist()
        
        current_datetime = ensure_timezone_aware(current_datetime)
        last_review_aware = ensure_timezone_aware(topic.last_review)
        
        elapsed_days = max(0, (current_datetime - last_review_aware).days)
        
        retrievability = (1 + (self._FACTOR * elapsed_days) / topic.stability) ** self._DECAY
        return max(0.0, min(1.0, retrievability))
    
    def review_topic(self, topic, rating, review_datetime=None, 
                    retention_percentage=None, review_duration=None):
        """
        Review a topic with given rating, updating its FSRS state
        Returns: (updated_topic, review_log)
        """
        if review_datetime is None:
            review_datetime = now_ist()
        else:
            review_datetime = ensure_timezone_aware(review_datetime)
        
        # Store state before review for logging
        stability_before = topic.stability
        difficulty_before = topic.difficulty
        
        # Calculate days since last review
        days_since_last_review = None
        if topic.last_review:
            last_review_aware = ensure_timezone_aware(topic.last_review)
            days_since_last_review = (review_datetime - last_review_aware).days
        
        # Update stability and difficulty based on current state
        if topic.state == State.Learning:
            self._process_learning_state(topic, rating, days_since_last_review, review_datetime)
        elif topic.state == State.Review:
            self._process_review_state(topic, rating, days_since_last_review, review_datetime)
        elif topic.state == State.Relearning:
            self._process_relearning_state(topic, rating, days_since_last_review, review_datetime)
        
        # Update topic's last review time
        topic.last_review = review_datetime
        topic.updated_at = review_datetime
                
        # Create review log
        review_log = TopicReviewLog(
            topic_id=topic.id,
            rating=rating,
            retention_percentage=retention_percentage,
            review_datetime=review_datetime,
            review_duration=review_duration,
            stability_before=stability_before,
            difficulty_before=difficulty_before,
            stability_after=topic.stability,
            difficulty_after=topic.difficulty
        )
        
        return topic, review_log

    def _process_learning_state(self, topic, rating, days_since_last_review, review_datetime):
        """Process topic in Learning state"""
        # Initialize stability and difficulty if first review
        if topic.stability is None and topic.difficulty is None:
            topic.stability = self._initial_stability(rating)
            topic.difficulty = self._initial_difficulty(rating, clamp=True)
        
        # Update parameters based on review timing
        elif days_since_last_review is not None and days_since_last_review < 1:
            # Same-day review
            topic.stability = self._short_term_stability(topic.stability, rating)
            topic.difficulty = self._next_difficulty(topic.difficulty, rating)
        else:
            # Multi-day review
            retrievability = self.get_topic_retrievability(topic, review_datetime)
            if rating == Rating.Again:
                topic.stability = self._next_forget_stability(
                    topic.difficulty, topic.stability, retrievability
                )
            else:
                topic.stability = self._next_recall_stability(
                    topic.difficulty, topic.stability, retrievability, rating
                )
            topic.difficulty = self._next_difficulty(topic.difficulty, rating)
        
        # Calculate next interval and state
        if len(self.learning_steps) == 0 or (
            topic.step is not None and topic.step >= len(self.learning_steps) and 
            rating in (Rating.Hard, Rating.Good, Rating.Easy)
        ):
            # Graduate to Review state
            topic.state = State.Review
            topic.step = None
            next_interval_days = self._next_interval(topic.stability)
            next_interval = timedelta(days=next_interval_days)
        else:
            # Stay in Learning state
            next_interval = self._calculate_learning_interval(topic, rating)
        
        # Apply fuzzing if in Review state
        if topic.state == State.Review and self.enable_fuzzing:
            fuzzed_days = self._get_fuzzed_interval(next_interval.days)
            next_interval = timedelta(days=fuzzed_days)
        
        topic.due = review_datetime + next_interval
    
    def _process_review_state(self, topic, rating, days_since_last_review, review_datetime):
        """Process topic in Review state"""
        # Update stability and difficulty
        if days_since_last_review is not None and days_since_last_review < 1:
            # Same-day review
            topic.stability = self._short_term_stability(topic.stability, rating)
        else:
            # Multi-day review
            retrievability = self.get_topic_retrievability(topic, review_datetime)
            if rating == Rating.Again:
                topic.stability = self._next_forget_stability(
                    topic.difficulty, topic.stability, retrievability
                )
            else:
                topic.stability = self._next_recall_stability(
                    topic.difficulty, topic.stability, retrievability, rating
                )
        
        topic.difficulty = self._next_difficulty(topic.difficulty, rating)
        
        # Calculate next interval and state
        if rating == Rating.Again:
            if len(self.relearning_steps) == 0:
                # No relearning steps, stay in Review
                next_interval_days = self._next_interval(topic.stability)
                next_interval = timedelta(days=next_interval_days)
            else:
                # Enter Relearning state
                topic.state = State.Relearning
                topic.step = 0
                next_interval = self.relearning_steps[0]
        else:
            # Continue in Review state
            next_interval_days = self._next_interval(topic.stability)
            next_interval = timedelta(days=next_interval_days)
            
            # Apply fuzzing
            if self.enable_fuzzing:
                fuzzed_days = self._get_fuzzed_interval(next_interval.days)
                next_interval = timedelta(days=fuzzed_days)
        
        topic.due = review_datetime + next_interval
    
    def _process_relearning_state(self, topic, rating, days_since_last_review, review_datetime):
        """Process topic in Relearning state"""
        # Update stability and difficulty
        if days_since_last_review is not None and days_since_last_review < 1:
            # Same-day review
            topic.stability = self._short_term_stability(topic.stability, rating)
            topic.difficulty = self._next_difficulty(topic.difficulty, rating)
        else:
            # Multi-day review
            retrievability = self.get_topic_retrievability(topic, review_datetime)
            if rating == Rating.Again:
                topic.stability = self._next_forget_stability(
                    topic.difficulty, topic.stability, retrievability
                )
            else:
                topic.stability = self._next_recall_stability(
                    topic.difficulty, topic.stability, retrievability, rating
                )
            topic.difficulty = self._next_difficulty(topic.difficulty, rating)
        
        # Calculate next interval and state
        if len(self.relearning_steps) == 0 or (
            topic.step is not None and topic.step >= len(self.relearning_steps) and 
            rating in (Rating.Hard, Rating.Good, Rating.Easy)
        ):
            # Graduate back to Review state
            topic.state = State.Review
            topic.step = None
            next_interval_days = self._next_interval(topic.stability)
            next_interval = timedelta(days=next_interval_days)
            
            # Apply fuzzing
            if self.enable_fuzzing:
                fuzzed_days = self._get_fuzzed_interval(next_interval.days)
                next_interval = timedelta(days=fuzzed_days)
        else:
            # Stay in Relearning state
            next_interval = self._calculate_relearning_interval(topic, rating)
        
        topic.due = review_datetime + next_interval
    
    def _calculate_learning_interval(self, topic, rating):
        """Calculate interval for topic in Learning state"""
        if rating == Rating.Again:
            topic.step = 0
            return self.learning_steps[0]
        elif rating == Rating.Hard:
            if topic.step == 0 and len(self.learning_steps) == 1:
                return self.learning_steps[0] * 1.5
            elif topic.step == 0 and len(self.learning_steps) >= 2:
                return (self.learning_steps[0] + self.learning_steps[1]) / 2.0
            else:
                return self.learning_steps[topic.step]
        elif rating == Rating.Good:
            if topic.step is None or topic.step + 1 >= len(self.learning_steps):
                # Graduate to Review state
                topic.state = State.Review
                topic.step = None
                next_interval_days = self._next_interval(topic.stability)
                return timedelta(days=next_interval_days)
            else:
                topic.step += 1
                return self.learning_steps[topic.step]
        elif rating == Rating.Easy:
            # Graduate to Review state
            topic.state = State.Review
            topic.step = None
            next_interval_days = self._next_interval(topic.stability)
            return timedelta(days=next_interval_days)
    
    def _calculate_relearning_interval(self, topic, rating):
        """Calculate interval for topic in Relearning state"""
        if rating == Rating.Again:
            topic.step = 0
            return self.relearning_steps[0]
        elif rating == Rating.Hard:
            if topic.step == 0 and len(self.relearning_steps) == 1:
                return self.relearning_steps[0] * 1.5
            elif topic.step == 0 and len(self.relearning_steps) >= 2:
                return (self.relearning_steps[0] + self.relearning_steps[1]) / 2.0
            else:
                return self.relearning_steps[topic.step]
        elif rating == Rating.Good:
            if topic.step is None or topic.step + 1 >= len(self.relearning_steps):
                # Graduate back to Review state
                topic.state = State.Review
                topic.step = None
                next_interval_days = self._next_interval(topic.stability)
                return timedelta(days=next_interval_days)
            else:
                topic.step += 1
                return self.relearning_steps[topic.step]
        elif rating == Rating.Easy:
            # Graduate back to Review state
            topic.state = State.Review
            topic.step = None
            next_interval_days = self._next_interval(topic.stability)
            return timedelta(days=next_interval_days)
    
    def get_priority_topics(self, limit=3, current_datetime=None):
        """
        Get top priority topics for review based on FSRS + exam urgency
        Returns topics sorted by priority score (highest first)
        """
        if current_datetime is None:
            current_datetime = now_ist()
        else:
            current_datetime = ensure_timezone_aware(current_datetime)
        
        # Get all topics
        all_topics = Topic.query.all()
        print(f"Total topics in database: {len(all_topics)}")
        
        # Filter topics that are due or overdue
        due_topics = []
        for topic in all_topics:
            topic_due = ensure_timezone_aware(topic.due) if topic.due else current_datetime
            if topic_due <= current_datetime:
                due_topics.append(topic)
                print(f"Topic {topic.name} is due: {topic_due} <= {current_datetime}")
            else:
                print(f"Topic {topic.name} not due: {topic_due} > {current_datetime}")
    
        print(f"Due topics: {len(due_topics)}")
        
        # If no topics are due, get the next few topics that will be due soon
        if not due_topics:
            print("No due topics, getting next upcoming topics")
            all_topics_sorted = sorted(all_topics, key=lambda t: ensure_timezone_aware(t.due) if t.due else current_datetime)
            due_topics = all_topics_sorted[:limit]
        
        # Calculate priority scores
        topic_priorities = []
        for topic in due_topics:
            priority_score = topic.calculate_priority_score(current_datetime, self.parameters)
            topic_priorities.append((topic, priority_score))
            print(f"Topic: {topic.name}, Priority Score: {priority_score}")
        
        # Sort by priority score (descending) and return top N
        topic_priorities.sort(key=lambda x: x[1], reverse=True)
        return [topic for topic, score in topic_priorities[:limit]]

    def to_dict(self):
        """Convert scheduler to dictionary for serialization"""
        return {
            'parameters': list(self.parameters),
            'desired_retention': self.desired_retention,
            'learning_steps': [int(step.total_seconds()) for step in self.learning_steps],
            'relearning_steps': [int(step.total_seconds()) for step in self.relearning_steps],
            'maximum_interval': self.maximum_interval,
            'enable_fuzzing': self.enable_fuzzing,
        }
    
    @staticmethod
    def from_dict(source_dict):
        """Create scheduler from dictionary"""
        return TopicScheduler(
            parameters=source_dict['parameters'],
            desired_retention=source_dict['desired_retention'],
            learning_steps=[timedelta(seconds=s) for s in source_dict['learning_steps']],
            relearning_steps=[timedelta(seconds=s) for s in source_dict['relearning_steps']],
            maximum_interval=source_dict['maximum_interval'],
            enable_fuzzing=source_dict['enable_fuzzing'],
        )