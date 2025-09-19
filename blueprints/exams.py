from flask import Blueprint, render_template, request, redirect, url_for, flash
from datetime import datetime, date, timedelta
from models import db, Topic, TopicExamDate, TopicReviewLog, now_ist
from services.exam import calculate_exam_preparation_strength
import statistics

exams_bp = Blueprint('exams', __name__, url_prefix='/exams')

@exams_bp.route('/')
def list_exams():
    """List all exams with preparation status"""
    # Get unique exams
    exam_data_raw = db.session.query(
        TopicExamDate.exam_name,
        TopicExamDate.exam_date,
        db.func.count(TopicExamDate.topic_id).label('topic_count')
    ).group_by(TopicExamDate.exam_name, TopicExamDate.exam_date)\
     .order_by(TopicExamDate.exam_date.asc()).all()
    
    current_date = date.today()
    exam_list = []
    
    for exam_name, exam_date, topic_count in exam_data_raw:
        days_until_exam = (exam_date - current_date).days
        
        # Determine status
        if days_until_exam < 0:
            status = "completed"
            status_class = "secondary"
        elif days_until_exam == 0:
            status = "today"
            status_class = "warning"
        elif days_until_exam <= 3:
            status = "urgent"
            status_class = "danger"
        elif days_until_exam <= 7:
            status = "soon"
            status_class = "warning"
        elif days_until_exam <= 30:
            status = "upcoming"
            status_class = "info"
        else:
            status = "distant"
            status_class = "light"
        
        # Get a sample exam date record for detailed analysis
        sample_exam = TopicExamDate.query.filter_by(
            exam_name=exam_name, exam_date=exam_date
        ).first()
        
        # Quick readiness calculation
        exam_topics = db.session.query(Topic).join(TopicExamDate).filter(
            TopicExamDate.exam_name == exam_name,
            TopicExamDate.exam_date == exam_date
        ).all()
        
        ready_count = 0
        total_retrievability = 0
        
        for topic in exam_topics:
            retrievability = topic.get_current_retrievability()
            total_retrievability += retrievability
            if retrievability >= 0.8:  # 80% threshold for "ready"
                ready_count += 1
        
        avg_readiness = (total_retrievability / topic_count * 100) if topic_count > 0 else 0
        readiness_percentage = (ready_count / topic_count * 100) if topic_count > 0 else 0
        
        exam_info = {
            'name': exam_name,
            'date': exam_date,
            'days_until': days_until_exam,
            'status': status,
            'status_class': status_class,
            'topic_count': topic_count,
            'ready_topics': ready_count,
            'readiness_percentage': round(readiness_percentage, 1),
            'avg_readiness': round(avg_readiness, 1)
        }
        exam_list.append(exam_info)
    
    # Separate into categories
    urgent_exams = [e for e in exam_list if e['status'] in ['today', 'urgent', 'soon']]
    upcoming_exams = [e for e in exam_list if e['status'] == 'upcoming']
    distant_exams = [e for e in exam_list if e['status'] == 'distant']
    completed_exams = [e for e in exam_list if e['status'] == 'completed']
    
    return render_template('exams/list.html',
                         urgent_exams=urgent_exams,
                         upcoming_exams=upcoming_exams,
                         distant_exams=distant_exams,
                         completed_exams=completed_exams)

@exams_bp.route('/<exam_name>/<exam_date>')
def view_exam(exam_name, exam_date):
    """View detailed exam preparation analysis"""
    try:
        exam_date_obj = datetime.strptime(exam_date, '%Y-%m-%d').date()
    except ValueError:
        flash('Invalid exam date format', 'error')
        return redirect(url_for('exams.list_exams'))
    
    # Get a sample exam record for analysis
    sample_exam = TopicExamDate.query.filter_by(
        exam_name=exam_name, exam_date=exam_date_obj
    ).first()
    
    if not sample_exam:
        flash('Exam not found', 'error')
        return redirect(url_for('exams.list_exams'))
    
    # Get comprehensive analysis
    analysis = calculate_exam_preparation_strength(sample_exam)
    
    # Get all topics for this exam with detailed info
    exam_topics = db.session.query(Topic).join(TopicExamDate).filter(
        TopicExamDate.exam_name == exam_name,
        TopicExamDate.exam_date == exam_date_obj
    ).all()
    
    # Calculate study recommendations
    weak_topics = [
        topic for topic in exam_topics 
        if topic.get_current_retrievability() < 0.7
    ]
    
    overdue_topics = [
        topic for topic in exam_topics
        if topic.due and topic.due < now_ist()
    ]
    
    # Performance insights
    all_reviews = []
    for topic in exam_topics:
        topic_reviews = TopicReviewLog.query.filter_by(topic_id=topic.id).all()
        all_reviews.extend(topic_reviews)
    
    recent_performance = None
    if all_reviews:
        # Last 2 weeks performance
        two_weeks_ago = now_ist() - timedelta(days=14)
        recent_reviews = [
            r for r in all_reviews 
            if r.review_datetime >= two_weeks_ago
        ]
        if recent_reviews:
            recent_performance = statistics.mean([r.rating for r in recent_reviews])
    
    return render_template('exams/view.html',
                         exam_name=exam_name,
                         exam_date=exam_date_obj,
                         analysis=analysis,
                         exam_topics=exam_topics,
                         weak_topics=weak_topics,
                         overdue_topics=overdue_topics,
                         recent_performance=round(recent_performance, 2) if recent_performance else None)

@exams_bp.route('/create', methods=['GET', 'POST'])
def create_exam():
    """Create a new exam and associate topics"""
    if request.method == 'POST':
        exam_name = request.form.get('exam_name', '').strip()
        exam_date_str = request.form.get('exam_date', '').strip()
        selected_topics = request.form.getlist('topic_ids')
        
        if not exam_name or not exam_date_str:
            flash('Exam name and date are required', 'error')
            return redirect(url_for('exams.create_exam'))
        
        try:
            exam_date_obj = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format', 'error')
            return redirect(url_for('exams.create_exam'))
        
        if not selected_topics:
            flash('Please select at least one topic', 'error')
            return redirect(url_for('exams.create_exam'))
        
        try:
            # Create exam associations
            created_count = 0
            for topic_id in selected_topics:
                if topic_id.strip():
                    # Check if association already exists
                    existing = TopicExamDate.query.filter_by(
                        topic_id=int(topic_id),
                        exam_name=exam_name,
                        exam_date=exam_date_obj
                    ).first()
                    
                    if not existing:
                        exam_association = TopicExamDate(
                            topic_id=int(topic_id),
                            exam_name=exam_name,
                            exam_date=exam_date_obj,
                            exam_weight=1.0
                        )
                        db.session.add(exam_association)
                        created_count += 1
            
            if created_count > 0:
                db.session.commit()
                flash(f'Exam "{exam_name}" created with {created_count} topics', 'success')
                return redirect(url_for('exams.view_exam', 
                               exam_name=exam_name, 
                               exam_date=exam_date_str))
            else:
                flash('All selected topics are already associated with this exam', 'warning')
                
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating exam: {str(e)}', 'error')
    
    # GET request - show form
    topics = Topic.query.order_by(Topic.subject, Topic.name).all()
    
    # Group topics by subject for better organization
    topics_by_subject = {}
    for topic in topics:
        subject = topic.subject or 'General'
        if subject not in topics_by_subject:
            topics_by_subject[subject] = []
        topics_by_subject[subject].append(topic)
    
    return render_template('exams/create.html', topics_by_subject=topics_by_subject)

@exams_bp.route('/<exam_name>/<exam_date>/study-plan')
def study_plan(exam_name, exam_date):
    """Generate a study plan for the exam"""
    try:
        exam_date_obj = datetime.strptime(exam_date, '%Y-%m-%d').date()
    except ValueError:
        flash('Invalid exam date format', 'error')
        return redirect(url_for('exams.list_exams'))
    
    sample_exam = TopicExamDate.query.filter_by(
        exam_name=exam_name, exam_date=exam_date_obj
    ).first()
    
    if not sample_exam:
        flash('Exam not found', 'error')
        return redirect(url_for('exams.list_exams'))
    
    # Get analysis
    analysis = calculate_exam_preparation_strength(sample_exam)
    
    # Create study schedule
    days_until_exam = analysis['days_to_exam']
    weak_topics = [t for t in analysis['topics_analysis'] if t['retrievability'] < 70]
    
    if days_until_exam <= 0:
        flash('This exam date has passed', 'info')
        return redirect(url_for('exams.view_exam', 
                       exam_name=exam_name, exam_date=exam_date))
    
    # Generate daily study plan
    study_schedule = []
    topics_per_day = max(1, len(weak_topics) // min(days_until_exam, 14))  # Max 14 days planning
    
    for day in range(min(days_until_exam, 14)):
        study_date = date.today() + timedelta(days=day)
        day_topics = weak_topics[day * topics_per_day:(day + 1) * topics_per_day]
        
        if day_topics or day == 0:  # Always show at least today
            study_schedule.append({
                'date': study_date,
                'day_name': study_date.strftime('%A'),
                'topics': day_topics,
                'focus': 'Review weak areas' if day_topics else 'General revision'
            })
    
    return render_template('exams/study_plan.html',
                         exam_name=exam_name,
                         exam_date=exam_date_obj,
                         analysis=analysis,
                         study_schedule=study_schedule,
                         days_until_exam=days_until_exam)

@exams_bp.route('/<exam_name>/<exam_date>/archive', methods=['POST'])
def archive_exam(exam_name, exam_date):
    """Archive an exam (remove all topic associations)"""
    try:
        exam_date_obj = datetime.strptime(exam_date, '%Y-%m-%d').date()
    except ValueError:
        flash('Invalid exam date format', 'error')
        return redirect(url_for('exams.list_exams'))
    
    try:
        # Delete all associations for this exam
        associations = TopicExamDate.query.filter_by(
            exam_name=exam_name, exam_date=exam_date_obj
        ).all()
        
        for association in associations:
            db.session.delete(association)
        
        db.session.commit()
        flash(f'Exam "{exam_name}" archived successfully', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error archiving exam: {str(e)}', 'error')
    
    return redirect(url_for('exams.list_exams'))