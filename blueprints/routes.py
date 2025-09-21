from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime, date, timedelta
from models import (
    db, Topic, ReviewSession, Exam, ExamTopic, StudySession, 
    ReviewRating, get_due_topics, get_upcoming_exams, 
    process_topic_review, get_learning_dashboard
)
from services.algorithm import ComprehensiveTopicScheduler
from utils.datetime_utils import now_ist,get_today
import math

bp = Blueprint('bp',__name__)

# ============================================================================
# DASHBOARD ROUTE
# ============================================================================

@bp.route('/')
def dashboard():
    """Main dashboard showing top 3 topics to review and closest exam"""
    
    # Get top 3 topics that need review (using algorithm priority)
    scheduler = ComprehensiveTopicScheduler()
    due_topics = get_due_topics(limit=10)  # Get more to calculate priorities
    # Calculate priority scores for each topic
    topic_priorities = []
    for topic in due_topics:
        memory = topic.to_algorithm_memory()
        strength = scheduler.calculate_realistic_topic_strength(memory)
        
        # Priority calculation: urgency + forgetting risk + exam proximity
        days_overdue = max(0, (datetime.now() - topic.next_review_date).days)
        urgency = min(1.0, days_overdue / 7.0)  # Normalize to week
        forgetting_risk = 1.0 - strength['exam_adjusted_retrievability']
        
        # Exam urgency bonus
        exam_bonus = 0.0
        if topic.exam_associations.count() > 0:
            nearest_exam_days = min([
                (assoc.exam.exam_date - date.today()).days 
                for assoc in topic.exam_associations 
                if assoc.exam.exam_date >= date.today()
            ] + [999])  # Default high value if no upcoming exams
            
            if nearest_exam_days <= 14:
                exam_bonus = (14 - nearest_exam_days) / 14.0
        
        priority_score = urgency * 0.4 + forgetting_risk * 0.4 + exam_bonus * 0.2
        
        topic_priorities.append({
            'topic': topic,
            'priority_score': priority_score,
            'retention_percentage': round(strength['exam_adjusted_retrievability'] * 100, 1),
            'days_overdue': days_overdue,
            'nearest_exam_days': nearest_exam_days if exam_bonus > 0 else None
        })
    
    # Sort by priority and take top 3
    topic_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
    top_topics = topic_priorities[:3]
    
    # Get closest upcoming exam
    upcoming_exams = get_upcoming_exams(days_ahead=90)
    closest_exam = None
    exam_readiness = 0.0
    
    if upcoming_exams:
        closest_exam = upcoming_exams[0]
        prep_summary = closest_exam.get_preparation_summary()
        exam_readiness = prep_summary['overall_readiness']
    
    # Get all topics for dropdown in review form
    all_topics = Topic.query.order_by(Topic.subject, Topic.name).all()
    
    return render_template('dashboard.html',
                         top_topics=top_topics,
                         closest_exam=closest_exam,
                         exam_readiness=exam_readiness,
                         all_topics=all_topics,today=get_today())


@bp.route('/log_review', methods=['POST'])
def log_review():
    """Handle review submission from dashboard form"""
    try:
        topic_id = int(request.form.get('topic_id'))
        rating = int(request.form.get('rating'))
        retention_percentage = request.form.get('retention_percentage')
        duration_minutes = request.form.get('duration_minutes')
        
        # Convert empty strings to None
        retention_percentage = float(retention_percentage) if retention_percentage else None
        duration_minutes = int(duration_minutes) if duration_minutes else None
        
        # Validate rating
        if rating not in [r.value for r in ReviewRating]:
            flash('Invalid rating provided', 'error')
            return redirect(url_for('bp.dashboard'))
        
        # Process the review using our algorithm
        result = process_topic_review(
            topic_id=topic_id,
            rating=rating,
            retention_percentage=retention_percentage,
            duration_minutes=duration_minutes,
            response_time_seconds=None,  # Not captured in this form
            study_context={'source': 'dashboard_quick_review'}
        )
        
        if result['success']:
            topic_name = Topic.query.get(topic_id).name
            flash(f'Review logged for {topic_name}!', 'success')
        else:
            flash('Error logging review. Please try again.', 'error')
            
    except (ValueError, TypeError) as e:
        flash('Invalid form data. Please check your inputs.', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    
    return redirect(url_for('bp.dashboard'))


# ============================================================================
# TOPICS ROUTES
# ============================================================================

@bp.route('/topics')
def topics_list():
    """List all topics with filtering and pagination"""
    
    # Get filter parameters
    subject_filter = request.args.get('subject', '').strip()
    mastery_filter = request.args.get('mastery', '').strip()
    search_query = request.args.get('search', '').strip()
    sort_by = request.args.get('sort', 'priority')
    page = int(request.args.get('page', 1))
    per_page = 20
    
    # Build query
    query = Topic.query
    
    if subject_filter:
        query = query.filter(Topic.subject == subject_filter)
    
    if mastery_filter:
        query = query.filter(Topic.mastery_level == mastery_filter)
    
    if search_query:
        query = query.filter(Topic.name.contains(search_query))
    
    # Get topics and calculate enhanced data
    topics = query.all()
    
    scheduler = ComprehensiveTopicScheduler()
    enhanced_topics = []
    
    for topic in topics:
        memory = topic.to_algorithm_memory()
        strength = scheduler.calculate_realistic_topic_strength(memory)
        
        days_until_due = (topic.next_review_date - datetime.now()).days if topic.next_review_date else 0
        
        enhanced_topics.append({
            'topic': topic,
            'strength': strength,
            'days_until_due': days_until_due,
            'is_overdue': days_until_due < 0,
            'priority_score': strength.get('exam_adjusted_retrievability', 0) * -1 + abs(min(0, days_until_due)) * 0.1
        })
    
    # Sort topics
    if sort_by == 'priority':
        enhanced_topics.sort(key=lambda x: x['priority_score'], reverse=True)
    elif sort_by == 'name':
        enhanced_topics.sort(key=lambda x: x['topic'].name.lower())
    elif sort_by == 'due_date':
        enhanced_topics.sort(key=lambda x: x['days_until_due'])
    elif sort_by == 'subject':
        enhanced_topics.sort(key=lambda x: x['topic'].subject or 'ZZZ')
    
    # Paginate
    total = len(enhanced_topics)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_topics = enhanced_topics[start:end]
    
    # Calculate pagination info
    total_pages = math.ceil(total / per_page)
    has_prev = page > 1
    has_next = page < total_pages
    
    # Get filter options
    subjects = db.session.query(Topic.subject).distinct().filter(Topic.subject.isnot(None)).all()
    subjects = [s[0] for s in subjects]
    
    mastery_levels = ['beginner', 'developing', 'proficient', 'advanced']
    
    return render_template('topics/list.html',
                         topics=paginated_topics,
                         subjects=subjects,
                         mastery_levels=mastery_levels,
                         current_filters={
                             'subject': subject_filter,
                             'mastery': mastery_filter,
                             'search': search_query,
                             'sort': sort_by
                         },
                         pagination={
                             'page': page,
                             'per_page': per_page,
                             'total': total,
                             'total_pages': total_pages,
                             'has_prev': has_prev,
                             'has_next': has_next
                         })


@bp.route('/topics/create', methods=['POST'])
def create_topic():
    """Create a new topic"""
    try:
        name = request.form.get('name', '').strip()
        subject = request.form.get('subject', '').strip()
        description = request.form.get('description', '').strip()
        complexity_rating = request.form.get('complexity_rating', 5.0)
        
        if not name:
            flash('Topic name is required', 'error')
            return redirect(url_for('bp.topics_list'))
        
        # Check for duplicates
        existing = Topic.query.filter(Topic.name.ilike(name)).first()
        if existing:
            flash(f'Topic "{name}" already exists', 'error')
            return redirect(url_for('bp.topics_list'))
        
        # Create new topic
        topic = Topic(
            name=name,
            subject=subject if subject else None,
            description=description if description else None,
            complexity_rating=float(complexity_rating)
        )
        
        db.session.add(topic)
        db.session.commit()
        
        flash(f'Topic "{name}" created successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating topic: {str(e)}', 'error')
    
    return redirect(url_for('bp.topics_list'))


@bp.route('/topics/<int:topic_id>')
def topic_detail(topic_id):
    """Detailed view of a specific topic"""
    topic = Topic.query.get_or_404(topic_id)
    
    # Get algorithm-based insights
    scheduler = ComprehensiveTopicScheduler()
    memory = topic.to_algorithm_memory()
    strength_analysis = scheduler.calculate_realistic_topic_strength(memory)
    progress_insights = topic.get_progress_insights()
    study_stats = topic.get_study_statistics()
    
    # Get recent performance with pagination
    page = int(request.args.get('page', 1))
    per_page = 10
    
    reviews_query = topic.review_sessions.order_by(ReviewSession.reviewed_at.desc())
    reviews_pagination = reviews_query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    recent_reviews = []
    for session in reviews_pagination.items:
        recent_reviews.append(session.to_user_display())
    
    # Get associated exams
    exam_associations = topic.exam_associations.join(Exam).filter(
        Exam.exam_date >= date.today()
    ).order_by(Exam.exam_date).all()
    
    upcoming_exams = []
    for assoc in exam_associations:
        days_left = (assoc.exam.exam_date - date.today()).days
        upcoming_exams.append({
            'exam': assoc.exam,
            'days_left': days_left,
            'importance': assoc.importance_weight,
            'expected_marks': assoc.expected_marks_percentage
        })
    
    return render_template('topics/detail.html',
                         topic=topic,
                         strength_analysis=strength_analysis,
                         progress_insights=progress_insights,
                         study_stats=study_stats,
                         recent_reviews=recent_reviews,
                         upcoming_exams=upcoming_exams,
                         reviews_pagination=reviews_pagination)


# ============================================================================
# EXAMS ROUTES
# ============================================================================

@bp.route('/exams/')
def exams_list():
    """List all exams with preparation status"""
    
    # Get all exams ordered by date
    all_exams = Exam.query.order_by(Exam.exam_date.asc()).all()
    
    current_date = date.today()
    categorized_exams = {
        'urgent': [],      # <= 7 days
        'upcoming': [],    # 8-30 days
        'future': [],      # > 30 days
        'completed': []    # past exams
    }
    
    for exam in all_exams:
        days_until = (exam.exam_date - current_date).days
        prep_summary = exam.get_preparation_summary()
        
        exam_info = {
            'exam': exam,
            'days_until': days_until,
            'prep_summary': prep_summary,
            'status_class': 'secondary'
        }
        
        if days_until < 0:
            exam_info['status_class'] = 'secondary'
            categorized_exams['completed'].append(exam_info)
        elif days_until <= 7:
            exam_info['status_class'] = 'danger' if prep_summary['overall_readiness'] < 70 else 'warning'
            categorized_exams['urgent'].append(exam_info)
        elif days_until <= 30:
            exam_info['status_class'] = 'warning' if prep_summary['overall_readiness'] < 60 else 'info'
            categorized_exams['upcoming'].append(exam_info)
        else:
            exam_info['status_class'] = 'light'
            categorized_exams['future'].append(exam_info)
    
    # Get all topics for creating new exams
    all_topics = Topic.query.order_by(Topic.subject, Topic.name).all()
    topics_by_subject = {}
    for topic in all_topics:
        subject = topic.subject or 'General'
        if subject not in topics_by_subject:
            topics_by_subject[subject] = []
        topics_by_subject[subject].append(topic)
    
    return render_template('/exams/list.html',
                         categorized_exams=categorized_exams,
                         topics_by_subject=topics_by_subject)


@bp.route('/exams/create', methods=['POST'])
def create_exam():
    """Create a new exam with associated topics"""
    try:
        exam_name = request.form.get('exam_name', '').strip()
        exam_date_str = request.form.get('exam_date', '').strip()
        description = request.form.get('description', '').strip()
        importance = request.form.get('importance', 'medium')
        exam_type = request.form.get('exam_type', '').strip()
        selected_topics = request.form.getlist('topic_ids')
        
        if not exam_name or not exam_date_str:
            flash('Exam name and date are required', 'error')
            return redirect(url_for('bp.exams_list'))
        
        try:
            exam_date_obj = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format', 'error')
            return redirect(url_for('bp.exams_list'))
        
        if not selected_topics:
            flash('Please select at least one topic', 'error')
            return redirect(url_for('bp.exams_list'))
        
        # Create exam
        exam = Exam(
            name=exam_name,
            exam_date=exam_date_obj,
            description=description if description else None,
            importance=importance,
            exam_type=exam_type if exam_type else None
        )
        
        db.session.add(exam)
        db.session.flush()  # Get exam ID
        
        # Create topic associations
        topics_added = 0
        for topic_id in selected_topics:
            if topic_id.strip():
                topic = Topic.query.get(int(topic_id))
                if topic:
                    exam_topic = ExamTopic(
                        exam_id=exam.id,
                        topic_id=topic.id,
                        importance_weight=1.0
                    )
                    db.session.add(exam_topic)
                    topics_added += 1
        
        db.session.commit()
        
        flash(f'Exam "{exam_name}" created with {topics_added} topics!', 'success')
        return redirect(url_for('bp.exam_detail', exam_id=exam.id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating exam: {str(e)}', 'error')
    
    return redirect(url_for('bp.exams_list'))


@bp.route('/exams/<int:exam_id>')
def exam_detail(exam_id):
    """Detailed view of exam preparation"""
    exam = Exam.query.get_or_404(exam_id)
    
    # Get comprehensive preparation analysis
    prep_summary = exam.get_preparation_summary()
    
    # Get topics associated with this exam (paginated)
    page = int(request.args.get('page', 1))
    per_page = 10
    
    topics_query = db.session.query(Topic, ExamTopic).join(ExamTopic).filter(
        ExamTopic.exam_id == exam_id
    ).order_by(Topic.name)
    
    # Calculate readiness for each topic
    scheduler = ComprehensiveTopicScheduler()
    
    # Get all topics first to calculate readiness
    all_topic_associations = topics_query.all()
    topics_with_readiness = []
    
    for topic, association in all_topic_associations:
        memory = topic.to_algorithm_memory()
        strength = scheduler.calculate_realistic_topic_strength(
            memory, exam_context={'overall_preparation': prep_summary['overall_readiness'] / 100}
        )
        
        topics_with_readiness.append({
            'topic': topic,
            'association': association,
            'readiness_score': strength['exam_adjusted_retrievability'] * 100,
            'readiness_category': strength['readiness_category'],
            'confidence_interval': strength['confidence_interval'],
            'last_reviewed': topic.last_reviewed_date.strftime('%b %d') if topic.last_reviewed_date else 'Never',
            'next_review': topic._format_next_review_timing(),
            'review_count': topic.total_reviews
        })
    
    # Sort by readiness (lowest first - most urgent)
    topics_with_readiness.sort(key=lambda x: x['readiness_score'])
    
    # Paginate
    total_topics = len(topics_with_readiness)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_topics = topics_with_readiness[start:end]
    
    # Calculate pagination info
    total_pages = math.ceil(total_topics / per_page)
    
    # Get recent reviews for all topics in this exam
    recent_reviews_query = ReviewSession.query.join(Topic).join(ExamTopic).filter(
        ExamTopic.exam_id == exam_id
    ).order_by(ReviewSession.reviewed_at.desc()).limit(20)
    
    recent_reviews = []
    for session in recent_reviews_query.all():
        recent_reviews.append({
            'session': session,
            'topic_name': session.topic.name,
            'display': session.to_user_display()
        })
    
    # Preparation strength distribution for pie chart
    readiness_counts = {
        'excellent': 0,
        'good': 0, 
        'fair': 0,
        'poor': 0,
        'critical': 0
    }
    
    for item in topics_with_readiness:
        readiness_counts[item['readiness_category']] += 1
    
    return render_template('exams/detail.html',
                         exam=exam,
                         prep_summary=prep_summary,
                         topics_with_readiness=paginated_topics,
                         recent_reviews=recent_reviews,
                         readiness_distribution=readiness_counts,
                         pagination={
                             'page': page,
                             'per_page': per_page,
                             'total': total_topics,
                             'total_pages': total_pages,
                             'has_prev': page > 1,
                             'has_next': page < total_pages
                         },
                         today=get_today())


@bp.route('/exams/<int:exam_id>/delete', methods=['POST'])
def delete_exam(exam_id):
    """Delete an exam and its associations"""
    try:
        exam = Exam.query.get_or_404(exam_id)
        exam_name = exam.name
        
        db.session.delete(exam)
        db.session.commit()
        
        flash(f'Exam "{exam_name}" deleted successfully', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting exam: {str(e)}', 'error')
    
    return redirect(url_for('bp.exams_list'))


# ============================================================================
# ANALYTICS ROUTE
# ============================================================================

@bp.route('/analytics')
def analytics():
    """Analytics dashboard with learning insights"""
    
    # Get dashboard data
    dashboard_data = get_learning_dashboard()
    
    # Get weekly progress for last 4 weeks
    weekly_stats = []
    for i in range(4):
        week_start = date.today() - timedelta(weeks=i, days=date.today().weekday())
        week_end = week_start + timedelta(days=6)
        
        # Count reviews in this week
        week_reviews = ReviewSession.query.filter(
            ReviewSession.reviewed_at >= datetime.combine(week_start, datetime.min.time()),
            ReviewSession.reviewed_at <= datetime.combine(week_end, datetime.max.time())
        ).all()
        
        successful_reviews = len([r for r in week_reviews if r.rating >= ReviewRating.GOOD])
        
        # Count study time
        week_study_time = db.session.query(
            db.func.sum(StudySession.duration_minutes)
        ).filter(
            StudySession.started_at >= datetime.combine(week_start, datetime.min.time()),
            StudySession.started_at <= datetime.combine(week_end, datetime.max.time())
        ).scalar() or 0
        
        weekly_stats.append({
            'week_start': week_start.strftime('%b %d'),
            'total_reviews': len(week_reviews),
            'successful_reviews': successful_reviews,
            'success_rate': round((successful_reviews / len(week_reviews)) * 100, 1) if week_reviews else 0,
            'study_hours': round(week_study_time / 60, 1)
        })
    
    weekly_stats.reverse()  # Show chronologically
    
    # Get subject-wise breakdown
    subjects_stats = []
    subjects = db.session.query(Topic.subject).distinct().filter(Topic.subject.isnot(None)).all()
    
    for (subject,) in subjects:
        subject_topics = Topic.query.filter(Topic.subject == subject).all()
        
        if subject_topics:
            scheduler = ComprehensiveTopicScheduler()
            readiness_scores = []
            
            for topic in subject_topics:
                memory = topic.to_algorithm_memory()
                strength = scheduler.calculate_realistic_topic_strength(memory)
                readiness_scores.append(strength['exam_adjusted_retrievability'])
            
            avg_readiness = sum(readiness_scores) / len(readiness_scores) * 100
            
            subjects_stats.append({
                'subject': subject,
                'topic_count': len(subject_topics),
                'avg_readiness': round(avg_readiness, 1),
                'status': 'excellent' if avg_readiness >= 85 else 
                         'good' if avg_readiness >= 70 else
                         'needs_work' if avg_readiness >= 55 else 'critical'
            })
    
    subjects_stats.sort(key=lambda x: x['avg_readiness'], reverse=True)
    
    return render_template('analytics.html',
                         dashboard_data=dashboard_data,
                         weekly_stats=weekly_stats,
                         subjects_stats=subjects_stats)


# ============================================================================
# API ENDPOINTS (for AJAX calls)
# ============================================================================

@bp.route('/api/topic/<int:topic_id>/quick_review', methods=['POST'])
def api_quick_review(topic_id):
    """API endpoint for quick topic reviews"""
    try:
        data = request.get_json()
        rating = int(data.get('rating'))
        
        result = process_topic_review(
            topic_id=topic_id,
            rating=rating,
            retention_percentage=data.get('retention_percentage'),
            duration_minutes=data.get('duration_minutes'),
            study_context={'source': 'quick_review_api'}
        )
        
        return jsonify({
            'success': True,
            'message': 'Review logged successfully',
            'next_review': result.get('next_review')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400