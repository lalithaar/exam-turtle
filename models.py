from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
import pytz
import math
# from copy import copy
# import json
from constants import *

ist = pytz.timezone('Asia/Kolkata')
db = SQLAlchemy()

def ensure_timezone_aware(dt, target_timezone=ist):
    """Ensure datetime object is timezone-aware and in IST"""
    if dt is None:
        return None
    
    if isinstance(dt, str):
        # If it's a string, parse it first
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
    if dt.tzinfo is not None:
        # Already timezone-aware, convert to IST
        return dt.astimezone(target_timezone)
    else:
        # Timezone-naive, assume it's in target timezone
        return target_timezone.localize(dt)

def now_ist():
    """Get current datetime in IST"""
    return datetime.now(ist)


class Topic(db.Model):
    __tablename__ = 'topics'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(100), nullable=True)
    
    # FSRS Core Parameters
    state = db.Column(db.Integer, default=State.Learning)
    step = db.Column(db.Integer, nullable=True)
    stability = db.Column(db.Float, nullable=True)
    difficulty = db.Column(db.Float, nullable=True)
    
    # Scheduling - Store as UTC, convert to IST when needed
    due = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_review = db.Column(db.DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    review_logs = db.relationship('TopicReviewLog', backref='topic', lazy=True, cascade='all, delete-orphan')
    exam_dates = db.relationship('TopicExamDate', backref='topic', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, name, subject=None, **kwargs):
        super().__init__()
        self.name = name
        self.subject = subject
        self.state = State.Learning
        self.step = 0 if self.state == State.Learning else None
        self.due = now_ist()
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_current_retrievability(self, current_datetime=None, scheduler_params=None):
        """Calculate current retrievability using FSRS forgetting curve"""
        if self.last_review is None or self.stability is None:
            return 0.0
            
        if current_datetime is None:
            current_datetime = now_ist()
        
        current_datetime = ensure_timezone_aware(current_datetime)
        last_review_aware = ensure_timezone_aware(self.last_review)

        if scheduler_params is None:
            scheduler_params = DEFAULT_PARAMETERS

        elapsed_days = max(0, (current_datetime - last_review_aware).days)

        # FSRS Forgetting Curve: R(t,S) = (1 + FACTOR × t/(9×S))^DECAY
        decay = -scheduler_params[20]  # w[20]
        factor = 0.9 ** (1 / decay) - 1
        
        retrievability = (1 + (factor * elapsed_days) / self.stability) ** decay
        return max(0.0, min(1.0, retrievability))
    
    def get_exam_urgency(self, current_datetime=None):
        """Calculate exam urgency factor based on upcoming exams"""
        if current_datetime is None:
            current_datetime = now_ist()
        
        current_datetime = ensure_timezone_aware(current_datetime)
            
        if not self.exam_dates:
            return 0.0
            
        # Find the nearest upcoming exam
        upcoming_exams = [
            exam for exam in self.exam_dates 
            if exam.exam_date >= current_datetime.date()
        ]
        
        if not upcoming_exams:
            return 0.0
            
        # Calculate urgency based on nearest exam (exponential decay)
        nearest_exam = min(upcoming_exams, key=lambda x: x.exam_date)
        days_to_exam = (nearest_exam.exam_date - current_datetime.date()).days
        
        # Exponential urgency: closer exams get exponentially higher priority
        urgency = math.exp(-max(0, days_to_exam) / 30.0)  # 30-day decay constant
        return urgency
    
    def calculate_priority_score(self, current_datetime=None, scheduler_params=None):
        """
        Calculate priority score combining FSRS retrievability and exam urgency
        Higher score = higher priority for review
        """
        retrievability = self.get_current_retrievability(current_datetime, scheduler_params)
        exam_urgency = self.get_exam_urgency(current_datetime)
        
        # Forgetting probability (1 - retrievability)
        forgetting_prob = 1 - retrievability
        
        # Combine forgetting probability with exam urgency
        priority_score = forgetting_prob * (1 + exam_urgency * 2.0)  # 2x multiplier for exam urgency
        
        return priority_score
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'subject': self.subject,
            'state': self.state,
            'step': self.step,
            'stability': self.stability,
            'difficulty': self.difficulty,
            'due': self.due.isoformat() if self.due else None,
            'last_review': self.last_review.isoformat() if self.last_review else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @staticmethod
    def from_dict(source_dict):
        """Create Topic from dictionary"""
        topic_data = source_dict.copy()
        
        # Convert datetime strings back to datetime objects
        for date_field in ['due', 'last_review', 'created_at', 'updated_at']:
            if topic_data.get(date_field):
                topic_data[date_field] = ensure_timezone_aware(topic_data[date_field])
        
        return Topic(**topic_data)

class TopicReviewLog(db.Model):
    __tablename__ = 'topic_review_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)
    
    rating = db.Column(db.Integer, nullable=False)
    retention_percentage = db.Column(db.Float, nullable=True)
    review_datetime = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    review_duration = db.Column(db.Integer, nullable=True)
    
    # FSRS state at time of review (for history tracking)
    stability_before = db.Column(db.Float, nullable=True)
    difficulty_before = db.Column(db.Float, nullable=True)
    stability_after = db.Column(db.Float, nullable=True)
    difficulty_after = db.Column(db.Float, nullable=True)
    
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __init__(self, topic_id, rating, retention_percentage=None, review_duration=None, **kwargs):
        super().__init__()
        self.topic_id = topic_id
        self.rating = rating
        self.retention_percentage = retention_percentage
        self.review_duration = review_duration
        self.review_datetime = kwargs.get('review_datetime', now_ist())
        
        # Store additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @staticmethod
    def convert_retention_to_rating(retention_percentage):
        """Convert retention percentage to FSRS rating"""
        if retention_percentage is None:
            return Rating.Good  # Default
        
        if retention_percentage < 60:
            return Rating.Again
        elif retention_percentage < 75:
            return Rating.Hard
        elif retention_percentage < 90:
            return Rating.Good
        else:
            return Rating.Easy
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'topic_id': self.topic_id,
            'rating': self.rating,
            'retention_percentage': self.retention_percentage,
            'review_datetime': self.review_datetime.isoformat(),
            'review_duration': self.review_duration,
            'stability_before': self.stability_before,
            'difficulty_before': self.difficulty_before,
            'stability_after': self.stability_after,
            'difficulty_after': self.difficulty_after,
            'created_at': self.created_at.isoformat(),
        }

    def to_print(self):
        review_dt = ensure_timezone_aware(self.review_datetime)
        created_dt = ensure_timezone_aware(self.created_at)
        
        return (f"""id: {self.id},
            topic_id: {self.topic_id},
            rating: {self.rating},
            retention_percentage: {self.retention_percentage},
            review_datetime: {review_dt.isoformat()},
            review_duration: {self.review_duration},
            stability_before: {self.stability_before},
            difficulty_before: {self.difficulty_before},
            stability_after: {self.stability_after},
            difficulty_after: {self.difficulty_after},
            created_at: {created_dt.isoformat()}""")

class TopicExamDate(db.Model):
    __tablename__ = 'topic_exam_dates'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)
    
    exam_name = db.Column(db.String(200), nullable=False)
    exam_date = db.Column(db.Date, nullable=False)
    exam_weight = db.Column(db.Float, default=1.0)
    
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __init__(self, topic_id, exam_name, exam_date, exam_weight=1.0):
        super().__init__()
        self.topic_id = topic_id
        self.exam_name = exam_name
        self.exam_date = exam_date
        self.exam_weight = exam_weight
    
    def to_dict(self):
        return {
            'id': self.id,
            'topic_id': self.topic_id,
            'exam_name': self.exam_name,
            'exam_date': self.exam_date.isoformat(),
            'exam_weight': self.exam_weight,
            'created_at': self.created_at.isoformat(),
        }
