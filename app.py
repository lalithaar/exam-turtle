from flask import Flask
from models import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

db.init_app(app)

# from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
# from datetime import datetime, timezone,timedelta
# from models import Topic, TopicReviewLog, TopicExamDate, TopicScheduler, Rating, ensure_timezone_aware, now_ist
# from algorithms import calculate_exam_progress

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     scheduler = TopicScheduler()
    
#     if request.method == 'POST':
#         # Handle form submission
#         topic_id = request.form.get('topic_id')
#         rating = int(request.form.get('rating'))
#         retention_percentage = request.form.get('retention_percentage')
#         review_duration = request.form.get('review_duration')
        
#         # Convert empty strings to None
#         retention_percentage = float(retention_percentage) if retention_percentage else None
#         review_duration = int(review_duration) if review_duration else None
        
#         # Get the topic and review it
#         topic = Topic.query.get_or_404(topic_id)
#         updated_topic, review_log = scheduler.review_topic(
#             topic=topic,
#             rating=rating,
#             retention_percentage=retention_percentage,
#             review_duration=review_duration
#         )
        
        
#         # Update the existing topic record
#         topic.state = updated_topic.state
#         topic.step = updated_topic.step
#         topic.stability = updated_topic.stability
#         topic.difficulty = updated_topic.difficulty
#         topic.due = updated_topic.due
#         topic.last_review = updated_topic.last_review
#         topic.updated_at = updated_topic.updated_at
        
#         # Add the review log
#         db.session.add(review_log)
        
#         try:
#             db.session.commit()
#             print('Successfully committed to database')
#             flash(f'Review logged for {topic.name}!', 'success')
#         except Exception as e:
#             db.session.rollback()
#             print(f'Error committing to database: {e}')
#             flash(f'Error logging review: {e}', 'error')
        
#         return redirect(url_for('index'))
    
#     # GET request - show the dashboard
#     current_datetime = now_ist()
#     priority_topics = scheduler.get_priority_topics(limit=3, current_datetime=current_datetime)
    
#     # Calculate additional info for display
#     topic_data = []
#     for topic in priority_topics:
#         retrievability = topic.get_current_retrievability(current_datetime, scheduler.parameters)
#         priority_score = topic.calculate_priority_score(current_datetime, scheduler.parameters)
#         days_since_last_review = (
#             (current_datetime - ensure_timezone_aware(topic.last_review)).days 
#             if topic.last_review else 0
#         )
        
#         topic_data.append({
#             'topic': topic,
#             'retrievability': round(retrievability * 100, 1),  # Convert to percentage
#             'priority_score': round(priority_score, 3),
#             'days_since_last_review': days_since_last_review
#         })
    
#     return render_template('index.html', topic_data=topic_data)

# @app.route('/manage')
# def manage():
#     """Route to manage topics and exam dates with modals"""
#     topics = Topic.query.order_by(Topic.name).all()
#     exam_dates = TopicExamDate.query.join(Topic).order_by(TopicExamDate.exam_date.desc()).all()
    
#     # Get unique exam names for the multiselect
#     unique_exams = db.session.query(TopicExamDate.exam_name).distinct().order_by(TopicExamDate.exam_name).all()
#     unique_exam_names = [exam[0] for exam in unique_exams]

#     ed = {}
#     for exam in exam_dates:
#         if exam.exam_name in ed.keys():
#             ed[exam.exam_name]['topics'].append(exam.topic.name)
#         else:
#             ed[exam.exam_name] = {
#                 'name': exam.exam_name,
#                 'date':exam.exam_date.strftime('%Y-%m-%d'),
#                 'weight':exam.exam_weight,
#                 'topics':[exam.topic.name,]
#             }    
#     return render_template('manage.html', 
#                          topics=topics, 
#                          ed=ed,
#                          unique_exam_names=unique_exam_names)

# @app.route('/add_topic', methods=['POST'])
# def add_topic():
#     """Add a new topic with optional exam associations"""
#     try:
#         name = request.form.get('topic_name', '').strip()
#         subject = request.form.get('subject', '').strip()
#         selected_exams = request.form.getlist('exam_names')
        
#         if not name:
#             flash('Topic name is required', 'error')
#             return redirect(url_for('manage'))
        
#         # Check if topic already exists
#         existing_topic = Topic.query.filter(Topic.name.ilike(name)).first()
#         if existing_topic:
#             flash(f'Topic "{name}" already exists', 'error')
#             return redirect(url_for('manage'))
        
#         # Create new topic
#         new_topic = Topic(name=name, subject=subject if subject else None)
#         db.session.add(new_topic)
#         db.session.flush()  # Get the ID without committing
        
#         # Associate with selected exams
#         associated_exams = []
#         for exam_name in selected_exams:
#             if exam_name.strip():
#                 # Find existing exam dates with this name
#                 existing_exam_dates = TopicExamDate.query.filter_by(exam_name=exam_name.strip()).all()
                
#                 for exam_date_record in existing_exam_dates:
#                     # Create association between topic and this exam date
#                     topic_exam_association = TopicExamDate(
#                         topic_id=new_topic.id,
#                         exam_name=exam_date_record.exam_name,
#                         exam_date=exam_date_record.exam_date,
#                         exam_weight=exam_date_record.exam_weight
#                     )
#                     db.session.add(topic_exam_association)
#                     associated_exams.append(exam_name.strip())
        
#         db.session.commit()
        
#         if associated_exams:
#             flash(f'Topic "{name}" created successfully and associated with exams: {", ".join(set(associated_exams))}', 'success')
#         else:
#             flash(f'Topic "{name}" created successfully', 'success')
            
#     except Exception as e:
#         db.session.rollback()
#         flash(f'Error creating topic: {str(e)}', 'error')
    
#     return redirect(url_for('manage'))

# @app.route('/add_exam_date', methods=['POST'])
# def add_exam_date():
#     """Add a new exam date"""
#     try:
#         exam_name = request.form.get('exam_name', '').strip()
#         exam_date_str = request.form.get('exam_date', '').strip()
#         exam_weight = request.form.get('exam_weight', '1.0').strip()
        
#         if not exam_name:
#             flash('Exam name is required', 'error')
#             return redirect(url_for('manage'))
        
#         if not exam_date_str:
#             flash('Exam date is required', 'error')
#             return redirect(url_for('manage'))
        
#         try:
#             exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
#         except ValueError:
#             flash('Invalid date format. Please use YYYY-MM-DD', 'error')
#             return redirect(url_for('manage'))
        
#         try:
#             exam_weight_float = float(exam_weight)
#             if exam_weight_float <= 0:
#                 raise ValueError("Weight must be positive")
#         except ValueError:
#             flash('Exam weight must be a positive number', 'error')
#             return redirect(url_for('manage'))
        
#         # Check if this exact exam already exists
#         existing_exam = TopicExamDate.query.filter_by(
#             exam_name=exam_name,
#             exam_date=exam_date
#         ).first()
        
#         if existing_exam:
#             flash(f'Exam "{exam_name}" on {exam_date} already exists', 'error')
#             return redirect(url_for('manage'))
        
#         # Create a placeholder topic-less exam date record
#         # This will be used as a template when associating with topics
#         new_exam_date = TopicExamDate(
#             topic_id=None,  # Will be set when associated with specific topics
#             exam_name=exam_name,
#             exam_date=exam_date,
#             exam_weight=exam_weight_float
#         )
        
#         # Instead, we'll create a record in a separate table or handle differently
#         # For now, let's create it without topic_id and handle it in the topic creation
#         # Actually, let's create it with the first topic's ID if any exist, or skip for now
        
#         # Better approach: Create exam dates associated with existing topics if requested
#         selected_topics = request.form.getlist('topic_ids')
        
#         if not selected_topics:
#             # Create a standalone exam date entry (we'll need to modify schema for this)
#             # For now, let's flash an error
#             flash('Please select at least one topic to associate this exam date with', 'error')
#             return redirect(url_for('manage'))
        
#         created_associations = []
#         for topic_id in selected_topics:
#             if topic_id.strip():
#                 try:
#                     topic_id_int = int(topic_id.strip())
#                     topic = Topic.query.get(topic_id_int)
                    
#                     if topic:
#                         # Check if this topic already has this exam
#                         existing_association = TopicExamDate.query.filter_by(
#                             topic_id=topic_id_int,
#                             exam_name=exam_name,
#                             exam_date=exam_date
#                         ).first()
                        
#                         if not existing_association:
#                             topic_exam_date = TopicExamDate(
#                                 topic_id=topic_id_int,
#                                 exam_name=exam_name,
#                                 exam_date=exam_date,
#                                 exam_weight=exam_weight_float
#                             )
#                             db.session.add(topic_exam_date)
#                             created_associations.append(topic.name)
                
#                 except ValueError:
#                     continue
        
#         if created_associations:
#             db.session.commit()
#             flash(f'Exam "{exam_name}" on {exam_date} created and associated with topics: {", ".join(created_associations)}', 'success')
#         else:
#             flash('No valid topics selected for exam date association', 'error')
            
#     except Exception as e:
#         db.session.rollback()
#         flash(f'Error creating exam date: {str(e)}', 'error')
    
#     return redirect(url_for('manage'))

# @app.route('/get_topics_json')
# def get_topics_json():
#     """API endpoint to get topics as JSON for dynamic loading"""
#     topics = Topic.query.order_by(Topic.name).all()
#     topics_data = [{'id': topic.id, 'name': topic.name, 'subject': topic.subject} for topic in topics]
#     return jsonify(topics_data)


# from sqlalchemy import func, case
# from flask import render_template, jsonify
# from sqlalchemy import func, case
# from datetime import datetime, timedelta
# import json


# @app.route('/statistics')
# def statistics():
#     """Main statistics dashboard"""
#     return render_template('statistics.html')

# @app.route('/api/topic-statistics')
# def topic_statistics():
#     """API endpoint for topic-wise statistics"""
#     scheduler = TopicScheduler()
#     current_datetime = now_ist()
    
#     topics = Topic.query.all()
#     topic_stats = []
    
#     for topic in topics:
#         # Calculate basic stats
#         total_reviews = TopicReviewLog.query.filter_by(topic_id=topic.id).count()
        
#         if total_reviews > 0:
#             # Average rating
#             avg_rating = db.session.query(func.avg(TopicReviewLog.rating)).filter_by(topic_id=topic.id).scalar()
            
#             # Recent performance (last 7 days)
#             week_ago = current_datetime - timedelta(days=7)
#             recent_reviews = TopicReviewLog.query.filter(
#                 TopicReviewLog.topic_id == topic.id,
#                 TopicReviewLog.review_datetime >= week_ago
#             ).all()
            
#             # Performance trend
#             if len(recent_reviews) >= 2:
#                 recent_avg = sum(r.rating for r in recent_reviews) / len(recent_reviews)
#                 trend = "improving" if recent_avg > avg_rating else "declining" if recent_avg < avg_rating else "stable"
#             else:
#                 trend = "stable"
            
#             # Mastery level based on stability and difficulty
#             mastery_score = 0
#             if topic.stability and topic.difficulty:
#                 # Higher stability = better retention, lower difficulty = easier
#                 mastery_score = min(100, (topic.stability / 30) * 100 * (11 - topic.difficulty) / 10)
            
#             # Current retrievability
#             retrievability = topic.get_current_retrievability(current_datetime, scheduler.parameters)
            
#             # Study streak (consecutive days with reviews)
#             study_streak = calculate_study_streak(topic.id, current_datetime)
            
#         else:
#             avg_rating = 0
#             trend = "new"
#             mastery_score = 0
#             retrievability = 0
#             study_streak = 0
        
#         # Days since last review
#         days_since_review = 0
#         if topic.last_review:
#             days_since_review = (current_datetime - ensure_timezone_aware(topic.last_review)).days
        
#         topic_stats.append({
#             'name': topic.name,
#             'subject': topic.subject or 'General',
#             'total_reviews': total_reviews,
#             'avg_rating': round(avg_rating, 1) if avg_rating else 0,
#             'mastery_score': round(mastery_score, 1),
#             'retrievability': round(retrievability * 100, 1),
#             'trend': trend,
#             'days_since_review': days_since_review,
#             'study_streak': study_streak,
#             'state': ['New', 'Learning', 'Review', 'Relearning'][topic.state] if hasattr(topic, 'state') else 'New'
#         })
    
#     return jsonify(topic_stats)

# @app.route('/api/exam-statistics')
# def exam_statistics():
#     """API endpoint for exam-wise statistics with readiness calculation"""
#     current_datetime = now_ist()
#     scheduler = TopicScheduler()
    
#     # Get all exams
#     exam_dates = db.session.query(TopicExamDate.exam_name, TopicExamDate.exam_date, TopicExamDate.exam_weight).distinct().all()
    
#     exam_stats = []
    
#     for exam_name, exam_date, exam_weight in exam_dates:
#         # Get all topics for this exam
#         exam_topics = db.session.query(Topic).join(TopicExamDate).filter(
#             TopicExamDate.exam_name == exam_name
#         ).all()
        
#         if not exam_topics:
#             continue
        
#         # Calculate exam readiness
#         total_topics = len(exam_topics)
#         ready_topics = 0
#         total_mastery = 0
#         total_retrievability = 0
#         weak_topics = []
#         strong_topics = []
        
#         for topic in exam_topics:
#             # Calculate mastery and retrievability
#             mastery_score = 0
#             if topic.stability and topic.difficulty:
#                 mastery_score = min(100, (topic.stability / 30) * 100 * (11 - topic.difficulty) / 10)
            
#             retrievability = topic.get_current_retrievability(current_datetime, scheduler.parameters) * 100
            
#             total_mastery += mastery_score
#             total_retrievability += retrievability
            
#             # Consider topic "ready" if mastery > 60% and retrievability > 70%
#             if mastery_score > 60 and retrievability > 70:
#                 ready_topics += 1
#                 strong_topics.append({
#                     'name': topic.name,
#                     'mastery': round(mastery_score, 1),
#                     'retrievability': round(retrievability, 1)
#                 })
#             else:
#                 weak_topics.append({
#                     'name': topic.name,
#                     'mastery': round(mastery_score, 1),
#                     'retrievability': round(retrievability, 1)
#                 })
        
#         # Calculate overall exam readiness
#         readiness_percentage = (ready_topics / total_topics) * 100 if total_topics > 0 else 0
#         avg_mastery = total_mastery / total_topics if total_topics > 0 else 0
#         avg_retrievability = total_retrievability / total_topics if total_topics > 0 else 0
        
#         # Days until exam
#         days_until_exam = (exam_date - current_datetime.date()).days if exam_date >= current_datetime.date() else 0
        
#         # Exam status
#         if days_until_exam < 0:
#             status = "completed"
#         elif days_until_exam == 0:
#             status = "today"
#         elif days_until_exam <= 7:
#             status = "urgent"
#         elif days_until_exam <= 30:
#             status = "upcoming"
#         else:
#             status = "distant"
        
#         # Recommendation
#         if readiness_percentage >= 80:
#             recommendation = "Well prepared! Focus on revision"
#         elif readiness_percentage >= 60:
#             recommendation = "Good progress. Review weak topics"
#         elif readiness_percentage >= 40:
#             recommendation = "Needs attention. Focus on weak areas"
#         else:
#             recommendation = "Critical! Intensive study required"
        
#         exam_stats.append({
#             'exam_name': exam_name,
#             'exam_date': exam_date.strftime('%Y-%m-%d'),
#             'days_until_exam': days_until_exam,
#             'status': status,
#             'total_topics': total_topics,
#             'ready_topics': ready_topics,
#             'readiness_percentage': round(readiness_percentage, 1),
#             'avg_mastery': round(avg_mastery, 1),
#             'avg_retrievability': round(avg_retrievability, 1),
#             'weak_topics': weak_topics[:5],  # Top 5 weak topics
#             'strong_topics': strong_topics[:5],  # Top 5 strong topics
#             'recommendation': recommendation,
#             'exam_weight': exam_weight
#         })
    
#     # Sort by exam date
#     exam_stats.sort(key=lambda x: x['exam_date'])
    
#     return jsonify(exam_stats)

# @app.route('/api/study-analytics')
# def study_analytics():
#     """API endpoint for overall study analytics"""
#     current_datetime = now_ist()
    
#     # Study activity over time (last 30 days)
#     thirty_days_ago = current_datetime - timedelta(days=30)
#     daily_reviews = db.session.query(
#         func.date(TopicReviewLog.review_datetime).label('date'),
#         func.count(TopicReviewLog.id).label('count'),
#         func.avg(TopicReviewLog.rating).label('avg_rating')
#     ).filter(
#         TopicReviewLog.review_datetime >= thirty_days_ago
#     ).group_by(
#         func.date(TopicReviewLog.review_datetime)
#     ).order_by('date').all()
    
#     # Subject-wise distribution
#     subject_stats = db.session.query(
#         Topic.subject,
#         func.count(Topic.id).label('topic_count'),
#         func.count(TopicReviewLog.id).label('review_count')
#     ).outerjoin(TopicReviewLog).group_by(Topic.subject).all()
    
#     # Rating distribution
#     rating_distribution = db.session.query(
#         TopicReviewLog.rating,
#         func.count(TopicReviewLog.id).label('count')
#     ).group_by(TopicReviewLog.rating).all()
    
#     return jsonify({
#         'daily_activity': [
#             {
#                 'date': str(date),
#                 'reviews': count,
#                 'avg_rating': round(float(avg_rating), 1) if avg_rating else 0
#             }
#             for date, count, avg_rating in daily_reviews
#         ],
#         'subject_distribution': [
#             {
#                 'subject': subject or 'General',
#                 'topics': topic_count,
#                 'reviews': review_count or 0
#             }
#             for subject, topic_count, review_count in subject_stats
#         ],
#         'rating_distribution': [
#             {
#                 'rating': rating,
#                 'count': count,
#                 'label': ['', 'Again', 'Hard', 'Good', 'Easy'][rating]
#             }
#             for rating, count in rating_distribution
#         ]
#     })

# def calculate_study_streak(topic_id, current_datetime):
#     """Calculate consecutive days with reviews for a topic"""
#     reviews = TopicReviewLog.query.filter_by(topic_id=topic_id).order_by(
#         TopicReviewLog.review_datetime.desc()
#     ).all()
    
#     if not reviews:
#         return 0
    
#     streak = 0
#     current_date = current_datetime.date()
    
#     # Group reviews by date
#     review_dates = set()
#     for review in reviews:
#         review_date = ensure_timezone_aware(review.review_datetime).date()
#         review_dates.add(review_date)
    
#     # Count consecutive days backwards from today
#     check_date = current_date
#     while check_date in review_dates:
#         streak += 1
#         check_date -= timedelta(days=1)
    
#     return streak
    
# register blueprints
from blueprints.dashboard import dashboard_bp
from blueprints.exams import exams_bp
from blueprints.topics import topics_bp

app.register_blueprint(dashboard_bp)
app.register_blueprint(exams_bp)
app.register_blueprint(topics_bp)


if __name__ == '__main__':
    
    app.run(debug=True)