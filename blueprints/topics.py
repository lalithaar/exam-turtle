from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import datetime, timedelta
from models import db, Topic, TopicReviewLog, TopicExamDate, now_ist, ensure_timezone_aware
from services.algorithm import TopicScheduler
from constants import DEFAULT_PARAMETERS, State, Rating

topics_bp = Blueprint('topics', __name__, url_prefix='/topics')

@topics_bp.route('/')
def list_topics():
    """List all topics with filters and search"""
    # Get filter parameters
    subject_filter = request.args.get('subject', '').strip()
    state_filter = request.args.get('state', '').strip()
    search_query = request.args.get('search', '').strip()
    sort_by = request.args.get('sort', 'priority')  # priority, name, due_date, subject
    
    # Base query
    query = Topic.query
    
    # Apply filters
    if subject_filter:
        query = query.filter(Topic.subject == subject_filter)
    
    if state_filter:
        try:
            state_int = int(state_filter)
            query = query.filter(Topic.state == state_int)
        except ValueError:
            pass
    
    if search_query:
        query = query.filter(Topic.name.contains(search_query))
    
    topics = query.all()
    
    # Calculate additional data for each topic
    current_datetime = now_ist()
    scheduler = TopicScheduler()
    
    topic_data = []
    for topic in topics:
        retrievability = topic.get_current_retrievability(current_datetime, DEFAULT_PARAMETERS)
        priority_score = topic.calculate_priority_score(current_datetime, DEFAULT_PARAMETERS)
        days_since_review = (
            (current_datetime - ensure_timezone_aware(topic.last_review)).days 
            if topic.last_review else None
        )
        days_until_due = (
            (ensure_timezone_aware(topic.due) - current_datetime).days 
            if topic.due else 0
        )
        
        # Get upcoming exams for this topic
        upcoming_exams = [
            exam for exam in topic.exam_dates 
            if exam.exam_date >= current_datetime.date()
        ]
        
        topic_info = {
            'topic': topic,
            'retrievability_percent': round(retrievability * 100, 1),
            'priority_score': round(priority_score, 3),
            'days_since_review': days_since_review,
            'days_until_due': days_until_due,
            'is_overdue': days_until_due < 0,
            'state_name': ['New', 'Learning', 'Review', 'Relearning'][topic.state],
            'upcoming_exams': upcoming_exams[:2],  # Show max 2 upcoming exams
            'total_reviews': TopicReviewLog.query.filter_by(topic_id=topic.id).count()
        }
        topic_data.append(topic_info)
    
    # Sort topics
    if sort_by == 'priority':
        topic_data.sort(key=lambda x: x['priority_score'], reverse=True)
    elif sort_by == 'name':
        topic_data.sort(key=lambda x: x['topic'].name.lower())
    elif sort_by == 'due_date':
        topic_data.sort(key=lambda x: x['days_until_due'])
    elif sort_by == 'subject':
        topic_data.sort(key=lambda x: x['topic'].subject or 'ZZZ')
    
    # Get unique subjects for filter dropdown
    unique_subjects = db.session.query(Topic.subject).distinct().all()
    subjects = [s[0] for s in unique_subjects if s[0]]
    
    # State options for filter
    states = [
        (State.Learning, 'Learning'),
        (State.Review, 'Review'),
        (State.Relearning, 'Relearning')
    ]
    
    return render_template('topics/list.html',
                         topic_data=topic_data,
                         subjects=subjects,
                         states=states,
                         current_filters={
                             'subject': subject_filter,
                             'state': state_filter,
                             'search': search_query,
                             'sort': sort_by
                         })

@topics_bp.route('/create', methods=['GET', 'POST'])
def create_topic():
    """Create a new topic"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        subject = request.form.get('subject', '').strip()
        
        if not name:
            flash('Topic name is required', 'error')
            return redirect(url_for('topics.create_topic'))
        
        # Check if topic already exists
        existing = Topic.query.filter(Topic.name.ilike(name)).first()
        if existing:
            flash(f'Topic "{name}" already exists', 'error')
            return redirect(url_for('topics.create_topic'))
        
        try:
            topic = Topic(name=name, subject=subject if subject else None)
            db.session.add(topic)
            db.session.commit()
            
            flash(f'Topic "{name}" created successfully', 'success')
            return redirect(url_for('topics.view_topic', id=topic.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating topic: {str(e)}', 'error')
    
    # GET request - show form
    # Get existing subjects for suggestions
    existing_subjects = db.session.query(Topic.subject).distinct().all()
    subjects = [s[0] for s in existing_subjects if s[0]]
    
    return render_template('topics/create.html', subjects=subjects)

@topics_bp.route('/<int:id>')
def view_topic(id):
    """View detailed topic information and review history"""
    topic = Topic.query.get_or_404(id)
    current_datetime = now_ist()
    
    # Calculate current stats
    retrievability = topic.get_current_retrievability(current_datetime, DEFAULT_PARAMETERS)
    priority_score = topic.calculate_priority_score(current_datetime, DEFAULT_PARAMETERS)
    
    # Get review history (last 20 reviews)
    recent_reviews = TopicReviewLog.query.filter_by(topic_id=id)\
        .order_by(TopicReviewLog.review_datetime.desc()).limit(20).all()
    
    # Calculate performance trends
    if len(recent_reviews) >= 5:
        recent_avg = sum(r.rating for r in recent_reviews[:5]) / 5
        older_avg = sum(r.rating for r in recent_reviews[5:10]) / min(5, len(recent_reviews) - 5) if len(recent_reviews) > 5 else recent_avg
        trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
    else:
        trend = "insufficient_data"
    
    # Get upcoming exams
    upcoming_exams = [
        exam for exam in topic.exam_dates 
        if exam.exam_date >= current_datetime.date()
    ]
    upcoming_exams.sort(key=lambda x: x.exam_date)
    
    # Calculate study streak
    study_streak = calculate_study_streak(topic, current_datetime)
    
    topic_stats = {
        'retrievability_percent': round(retrievability * 100, 1),
        'priority_score': round(priority_score, 3),
        'state_name': ['New', 'Learning', 'Review', 'Relearning'][topic.state],
        'total_reviews': len(topic.review_logs),
        'performance_trend': trend,
        'study_streak': study_streak,
        'days_since_review': (
            (current_datetime - ensure_timezone_aware(topic.last_review)).days 
            if topic.last_review else None
        ),
        'days_until_due': (
            (ensure_timezone_aware(topic.due) - current_datetime).days 
            if topic.due else 0
        )
    }
    
    return render_template('topics/view.html',
                         topic=topic,
                         topic_stats=topic_stats,
                         recent_reviews=recent_reviews,
                         upcoming_exams=upcoming_exams)

@topics_bp.route('/<int:id>/review', methods=['GET', 'POST'])
def review_topic(id):
    """Review a specific topic"""
    topic = Topic.query.get_or_404(id)
    
    if request.method == 'POST':
        try:
            rating = int(request.form.get('rating'))
            retention_percentage = request.form.get('retention_percentage')
            review_duration = request.form.get('review_duration')
            
            # Convert empty strings to None
            retention_percentage = float(retention_percentage) if retention_percentage else None
            review_duration = int(review_duration) if review_duration else None
            
            # Process the review
            scheduler = TopicScheduler()
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
            return redirect(url_for('topics.view_topic', id=id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error logging review: {str(e)}', 'error')
    
    # GET request - show review form
    current_datetime = now_ist()
    retrievability = topic.get_current_retrievability(current_datetime, DEFAULT_PARAMETERS)
    
    # Get recent performance for context
    recent_reviews = TopicReviewLog.query.filter_by(topic_id=id)\
        .order_by(TopicReviewLog.review_datetime.desc()).limit(5).all()
    
    return render_template('topics/review.html',
                         topic=topic,
                         retrievability_percent=round(retrievability * 100, 1),
                         recent_reviews=recent_reviews)

@topics_bp.route('/<int:id>/archive', methods=['POST'])
def archive_topic(id):
    """Archive a topic (soft delete)"""
    topic = Topic.query.get_or_404(id)
    
    try:
        # Instead of deleting, you could add an 'archived' field
        # For now, we'll actually delete but you can modify this
        topic_name = topic.name
        db.session.delete(topic)
        db.session.commit()
        
        flash(f'Topic "{topic_name}" archived successfully', 'success')
        return redirect(url_for('topics.list_topics'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error archiving topic: {str(e)}', 'error')
        return redirect(url_for('topics.view_topic', id=id))

def calculate_study_streak(topic, current_datetime):
    """Calculate consecutive days with reviews"""
    reviews = TopicReviewLog.query.filter_by(topic_id=topic.id)\
        .order_by(TopicReviewLog.review_datetime.desc()).all()
    
    if not reviews:
        return 0
    
    # Group by dates
    review_dates = set()
    for review in reviews:
        review_date = ensure_timezone_aware(review.review_datetime).date()
        review_dates.add(review_date)
    
    # Count consecutive days
    streak = 0
    check_date = current_datetime.date()
    
    while check_date in review_dates:
        streak += 1
        check_date -= timedelta(days=1)
    
    return streak