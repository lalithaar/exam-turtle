from models import TopicExamDate, Topic, TopicReviewLog
from services.algorithm import TopicScheduler
from constants import DEFAULT_PARAMETERS
from datetime import datetime, timedelta
import statistics

def calculate_exam_preparation_strength(exam_date: TopicExamDate):
    """
    Calculate overall preparation strength for a specific exam
    Returns a comprehensive analysis of readiness for all topics in the exam
    
    Args:
        exam_date: TopicExamDate object representing the exam
        
    Returns:
        dict: Comprehensive exam preparation analysis
    """
    
    # Get all topics associated with this exam
    # Find all topics that have this exam date
    all_exam_dates = TopicExamDate.query.filter_by(
        exam_name=exam_date.exam_name,
        exam_date=exam_date.exam_date
    ).all()
    
    if not all_exam_dates:
        return {
            'exam_name': exam_date.exam_name,
            'exam_date': exam_date.exam_date.isoformat(),
            'overall_strength': 0.0,
            'status': 'no_topics',
            'message': 'No topics found for this exam'
        }
    
    # Get all topics for this exam
    topic_ids = [ed.topic_id for ed in all_exam_dates]
    topics = Topic.query.filter(Topic.id.in_(topic_ids)).all()
    
    if not topics:
        return {
            'exam_name': exam_date.exam_name,
            'exam_date': exam_date.exam_date.isoformat(),
            'overall_strength': 0.0,
            'status': 'no_topics',
            'message': 'No topics found for this exam'
        }
    
    # Current datetime for calculations
    current_dt = datetime.now()
    days_to_exam = (exam_date.exam_date - current_dt.date()).days
    
    # Initialize scheduler for FSRS calculations
    scheduler = TopicScheduler()
    
    # Analyze each topic
    topic_analyses = []
    retrievability_scores = []
    priority_scores = []
    review_counts = []
    
    for topic in topics:
        # Get topic's exam weight (from the junction table)
        topic_exam_info = next(
            (ed for ed in all_exam_dates if ed.topic_id == topic.id), 
            None
        )
        exam_weight = topic_exam_info.exam_weight if topic_exam_info else 1.0
        
        # Calculate current retrievability using FSRS
        retrievability = topic.get_current_retrievability(
            current_datetime=current_dt,
            scheduler_params=DEFAULT_PARAMETERS
        )
        
        # Calculate priority score (includes exam urgency)
        priority_score = topic.calculate_priority_score(
            current_datetime=current_dt,
            scheduler_params=DEFAULT_PARAMETERS
        )
        
        # Calculate review statistics
        review_logs = TopicReviewLog.query.filter_by(topic_id=topic.id).all()
        total_reviews = len(review_logs)
        
        # Calculate recent performance (last 10 reviews or all if less)
        recent_reviews = review_logs[-10:] if len(review_logs) >= 10 else review_logs
        recent_performance = statistics.mean([log.rating for log in recent_reviews]) if recent_reviews else 0
        
        # Calculate days since last review
        days_since_review = 0
        if topic.last_review:
            days_since_review = (current_dt - topic.last_review.replace(tzinfo=None)).days
        
        # Determine topic readiness level
        if retrievability >= 0.9:
            readiness = "excellent"
        elif retrievability >= 0.8:
            readiness = "good"
        elif retrievability >= 0.7:
            readiness = "fair"
        elif retrievability >= 0.6:
            readiness = "poor"
        else:
            readiness = "critical"
        
        # Calculate topic strength score (weighted)
        # Factors: retrievability (40%), recent performance (30%), review frequency (20%), recency (10%)
        review_frequency_score = min(1.0, total_reviews / 10.0)  # Normalize to 10 reviews
        recency_score = max(0.0, 1.0 - (days_since_review / 30.0))  # Penalty after 30 days
        performance_score = recent_performance / 4.0 if recent_performance > 0 else 0  # Normalize rating to 0-1
        
        topic_strength = (
            retrievability * 0.4 +
            performance_score * 0.3 +
            review_frequency_score * 0.2 +
            recency_score * 0.1
        ) * exam_weight
        
        topic_analysis = {
            'topic_id': topic.id,
            'topic_name': topic.name,
            'subject': topic.subject,
            'retrievability': round(retrievability, 3),
            'priority_score': round(priority_score, 3),
            'readiness_level': readiness,
            'topic_strength': round(topic_strength, 3),
            'exam_weight': exam_weight,
            'total_reviews': total_reviews,
            'recent_performance': round(recent_performance, 2) if recent_performance > 0 else None,
            'days_since_review': days_since_review,
            'last_review_date': topic.last_review.date().isoformat() if topic.last_review else None,
            'due_date': topic.due.date().isoformat() if topic.due else None,
            'is_overdue': topic.due < current_dt if topic.due else False
        }
        
        topic_analyses.append(topic_analysis)
        retrievability_scores.append(retrievability * exam_weight)
        priority_scores.append(priority_score)
        review_counts.append(total_reviews)
    
    # Calculate overall exam preparation metrics
    total_weight = sum(ed.exam_weight for ed in all_exam_dates)
    weighted_avg_retrievability = sum(retrievability_scores) / total_weight if total_weight > 0 else 0
    avg_priority_score = statistics.mean(priority_scores) if priority_scores else 0
    total_topic_count = len(topics)
    
    # Count topics by readiness level
    readiness_distribution = {
        'excellent': len([t for t in topic_analyses if t['readiness_level'] == 'excellent']),
        'good': len([t for t in topic_analyses if t['readiness_level'] == 'good']),
        'fair': len([t for t in topic_analyses if t['readiness_level'] == 'fair']),
        'poor': len([t for t in topic_analyses if t['readiness_level'] == 'poor']),
        'critical': len([t for t in topic_analyses if t['readiness_level'] == 'critical'])
    }
    
    # Count overdue topics
    overdue_topics = [t for t in topic_analyses if t['is_overdue']]
    overdue_count = len(overdue_topics)
    
    # Calculate study recommendations
    high_priority_topics = [t for t in topic_analyses if t['priority_score'] > 0.5]
    topics_needing_review = [t for t in topic_analyses if t['retrievability'] < 0.8]
    
    # Determine overall exam readiness status
    if weighted_avg_retrievability >= 0.85 and overdue_count == 0:
        exam_status = "well_prepared"
        status_message = "Excellent preparation! You're ready for this exam."
    elif weighted_avg_retrievability >= 0.75 and overdue_count <= 2:
        exam_status = "mostly_prepared"
        status_message = "Good preparation with minor areas to review."
    elif weighted_avg_retrievability >= 0.65:
        exam_status = "needs_improvement"
        status_message = "Moderate preparation. Focus on weak topics."
    elif weighted_avg_retrievability >= 0.5:
        exam_status = "poor_preparation"
        status_message = "Poor preparation. Intensive study needed."
    else:
        exam_status = "critical"
        status_message = "Critical! Immediate and intensive study required."
    
    # Calculate study time recommendations
    if days_to_exam > 0:
        recommended_daily_reviews = min(20, max(5, len(topics_needing_review)))
        estimated_study_hours = len(topics_needing_review) * 0.5  # 30 minutes per weak topic
    else:
        recommended_daily_reviews = 0
        estimated_study_hours = 0
    
    return {
        'exam_name': exam_date.exam_name,
        'exam_date': exam_date.exam_date.isoformat(),
        'days_to_exam': days_to_exam,
        'overall_strength': round(weighted_avg_retrievability, 3),
        'exam_status': exam_status,
        'status_message': status_message,
        
        # Topic-level analysis
        'total_topics': total_topic_count,
        'topics_analysis': sorted(topic_analyses, key=lambda x: x['priority_score'], reverse=True),
        
        # Readiness distribution
        'readiness_distribution': readiness_distribution,
        'readiness_percentages': {
            level: round((count / total_topic_count) * 100, 1) 
            for level, count in readiness_distribution.items()
        },
        
        # Priority insights
        'high_priority_topics_count': len(high_priority_topics),
        'topics_needing_review_count': len(topics_needing_review),
        'overdue_topics_count': overdue_count,
        'overdue_topics': [t['topic_name'] for t in overdue_topics],
        
        # Study recommendations
        'recommendations': {
            'focus_topics': [t['topic_name'] for t in high_priority_topics[:5]],  # Top 5
            'daily_reviews_recommended': recommended_daily_reviews,
            'estimated_daily_study_hours': round(estimated_study_hours / max(1, days_to_exam), 1) if days_to_exam > 0 else 0,
            'total_estimated_study_hours': round(estimated_study_hours, 1)
        },
        
        # Statistical summary
        'statistics': {
            'avg_retrievability': round(weighted_avg_retrievability, 3),
            'avg_priority_score': round(avg_priority_score, 3),
            'avg_reviews_per_topic': round(statistics.mean(review_counts), 1) if review_counts else 0,
            'total_reviews_completed': sum(review_counts),
            'topics_reviewed_recently': len([t for t in topic_analyses if t['days_since_review'] <= 7])
        }
    }