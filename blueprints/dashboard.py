from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import datetime, timedelta
from models import db, Topic, TopicReviewLog, TopicExamDate, now_ist, ensure_timezone_aware
from services.algorithm import TopicScheduler
from constants import DEFAULT_PARAMETERS, USER_NAME

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET', 'POST'])
def index():
    """Minimalistic dashboard with integrated quick review form"""
    current_datetime = now_ist()
    scheduler = TopicScheduler()
    
    # Handle POST request - Quick review form submission
    if request.method == 'POST':
        try:
            topic_id = request.form.get('topic_id')
            rating = int(request.form.get('rating'))
            retention_percentage = request.form.get('retention_percentage')
            review_duration = request.form.get('review_duration')
            
            # Convert empty strings to None
            retention_percentage = float(retention_percentage) if retention_percentage else None
            review_duration = int(review_duration) if review_duration else None
            
            # Get the topic and review it
            topic = Topic.query.get_or_404(topic_id)
            updated_topic, review_log = scheduler.review_topic(
                topic=topic,
                rating=rating,
                retention_percentage=retention_percentage,
                review_duration=review_duration
            )
            
            # Update the existing topic record
            topic.state = updated_topic.state
            topic.step = updated_topic.step
            topic.stability = updated_topic.stability
            topic.difficulty = updated_topic.difficulty
            topic.due = updated_topic.due
            topic.last_review = updated_topic.last_review
            topic.updated_at = updated_topic.updated_at
            
            # Add the review log
            db.session.add(review_log)
            db.session.commit()
            
            flash(f'Review logged for {topic.name}!', 'success')
            return redirect(url_for('dashboard.index'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error logging review: {str(e)}', 'error')
            return redirect(url_for('dashboard.index'))
    
    # GET request - Build dashboard data
    # Get top 3 priority topics
    priority_topics = scheduler.get_priority_topics(limit=3, current_datetime=current_datetime)
    
    # Calculate minimal data for each topic
    topic_data = []
    for topic in priority_topics:
        retrievability = topic.get_current_retrievability(current_datetime, DEFAULT_PARAMETERS)
        
        # Get closest exam for this topic
        upcoming_exams = [
            exam for exam in topic.exam_dates 
            if exam.exam_date >= current_datetime.date()
        ]
        closest_exam = min(upcoming_exams, key=lambda x: x.exam_date) if upcoming_exams else None
        
        topic_data.append({
            'id': topic.id,  # Added ID for form
            'name': topic.name,
            'subject': topic.subject,
            'retrievability_percent': round(retrievability * 100, 1),
            'closest_exam': closest_exam.exam_name if closest_exam else None,
            'exam_date': closest_exam.exam_date if closest_exam else None,
            'days_to_exam': (closest_exam.exam_date - current_datetime.date()).days if closest_exam else None
        })
    
    # Find the single closest upcoming exam across all topics
    all_upcoming_exams = TopicExamDate.query.filter(
        TopicExamDate.exam_date >= current_datetime.date()
    ).all()
    
    closest_overall_exam = None
    if all_upcoming_exams:
        # Get the absolute closest exam
        closest_exam_record = min(all_upcoming_exams, key=lambda x: x.exam_date)
        
        # Calculate strength for this exam
        from services.exam import calculate_exam_preparation_strength
        exam_analysis = calculate_exam_preparation_strength(closest_exam_record)
        
        closest_overall_exam = {
            'name': closest_exam_record.exam_name,
            'date': closest_exam_record.exam_date,
            'days_away': (closest_exam_record.exam_date - current_datetime.date()).days,
            'strength_percent': round(exam_analysis['overall_strength'] * 100, 1)
        }
    
    # Get all topics for the quick review dropdown (in case user wants to review a different topic)
    all_topics = Topic.query.order_by(Topic.name).all()
    
    return render_template('dashboard.html',
                         user_name=USER_NAME,
                         topic_data=topic_data,
                         closest_exam=closest_overall_exam,
                         all_topics=all_topics)