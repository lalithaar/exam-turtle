from flask import Flask
from models import db
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

db.init_app(app)

from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
from datetime import datetime, timezone
from models import Topic, TopicReviewLog, TopicExamDate, TopicScheduler, Rating, ensure_timezone_aware, now_ist
# from algorithms import calculate_exam_progress

@app.route('/', methods=['GET', 'POST'])
def index():
    scheduler = TopicScheduler()
    
    if request.method == 'POST':
        # Handle form submission
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
        
        try:
            db.session.commit()
            print('Successfully committed to database')
            flash(f'Review logged for {topic.name}!', 'success')
        except Exception as e:
            db.session.rollback()
            print(f'Error committing to database: {e}')
            flash(f'Error logging review: {e}', 'error')
        
        return redirect(url_for('index'))
    
    # GET request - show the dashboard
    current_datetime = now_ist()
    priority_topics = scheduler.get_priority_topics(limit=3, current_datetime=current_datetime)
    
    # Calculate additional info for display
    topic_data = []
    for topic in priority_topics:
        retrievability = topic.get_current_retrievability(current_datetime, scheduler.parameters)
        priority_score = topic.calculate_priority_score(current_datetime, scheduler.parameters)
        days_since_last_review = (
            (current_datetime - ensure_timezone_aware(topic.last_review)).days 
            if topic.last_review else 0
        )
        
        topic_data.append({
            'topic': topic,
            'retrievability': round(retrievability * 100, 1),  # Convert to percentage
            'priority_score': round(priority_score, 3),
            'days_since_last_review': days_since_last_review
        })
    
    return render_template('index.html', topic_data=topic_data)

@app.route('/manage')
def manage():
    """Route to manage topics and exam dates with modals"""
    topics = Topic.query.order_by(Topic.name).all()
    exam_dates = TopicExamDate.query.join(Topic).order_by(TopicExamDate.exam_date.desc()).all()
    
    # Get unique exam names for the multiselect
    unique_exams = db.session.query(TopicExamDate.exam_name).distinct().order_by(TopicExamDate.exam_name).all()
    unique_exam_names = [exam[0] for exam in unique_exams]

    ed = {}
    for exam in exam_dates:
        if exam.exam_name in ed.keys():
            ed[exam.exam_name]['topics'].append(exam.topic.name)
        else:
            ed[exam.exam_name] = {
                'name': exam.exam_name,
                'date':exam.exam_date.strftime('%Y-%m-%d'),
                'weight':exam.exam_weight,
                'topics':[exam.topic.name,]
            }    
    return render_template('manage.html', 
                         topics=topics, 
                         ed=ed,
                         unique_exam_names=unique_exam_names)

@app.route('/add_topic', methods=['POST'])
def add_topic():
    """Add a new topic with optional exam associations"""
    try:
        name = request.form.get('topic_name', '').strip()
        subject = request.form.get('subject', '').strip()
        selected_exams = request.form.getlist('exam_names')
        
        if not name:
            flash('Topic name is required', 'error')
            return redirect(url_for('manage'))
        
        # Check if topic already exists
        existing_topic = Topic.query.filter(Topic.name.ilike(name)).first()
        if existing_topic:
            flash(f'Topic "{name}" already exists', 'error')
            return redirect(url_for('manage'))
        
        # Create new topic
        new_topic = Topic(name=name, subject=subject if subject else None)
        db.session.add(new_topic)
        db.session.flush()  # Get the ID without committing
        
        # Associate with selected exams
        associated_exams = []
        for exam_name in selected_exams:
            if exam_name.strip():
                # Find existing exam dates with this name
                existing_exam_dates = TopicExamDate.query.filter_by(exam_name=exam_name.strip()).all()
                
                for exam_date_record in existing_exam_dates:
                    # Create association between topic and this exam date
                    topic_exam_association = TopicExamDate(
                        topic_id=new_topic.id,
                        exam_name=exam_date_record.exam_name,
                        exam_date=exam_date_record.exam_date,
                        exam_weight=exam_date_record.exam_weight
                    )
                    db.session.add(topic_exam_association)
                    associated_exams.append(exam_name.strip())
        
        db.session.commit()
        
        if associated_exams:
            flash(f'Topic "{name}" created successfully and associated with exams: {", ".join(set(associated_exams))}', 'success')
        else:
            flash(f'Topic "{name}" created successfully', 'success')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating topic: {str(e)}', 'error')
    
    return redirect(url_for('manage'))

@app.route('/add_exam_date', methods=['POST'])
def add_exam_date():
    """Add a new exam date"""
    try:
        exam_name = request.form.get('exam_name', '').strip()
        exam_date_str = request.form.get('exam_date', '').strip()
        exam_weight = request.form.get('exam_weight', '1.0').strip()
        
        if not exam_name:
            flash('Exam name is required', 'error')
            return redirect(url_for('manage'))
        
        if not exam_date_str:
            flash('Exam date is required', 'error')
            return redirect(url_for('manage'))
        
        try:
            exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD', 'error')
            return redirect(url_for('manage'))
        
        try:
            exam_weight_float = float(exam_weight)
            if exam_weight_float <= 0:
                raise ValueError("Weight must be positive")
        except ValueError:
            flash('Exam weight must be a positive number', 'error')
            return redirect(url_for('manage'))
        
        # Check if this exact exam already exists
        existing_exam = TopicExamDate.query.filter_by(
            exam_name=exam_name,
            exam_date=exam_date
        ).first()
        
        if existing_exam:
            flash(f'Exam "{exam_name}" on {exam_date} already exists', 'error')
            return redirect(url_for('manage'))
        
        # Create a placeholder topic-less exam date record
        # This will be used as a template when associating with topics
        new_exam_date = TopicExamDate(
            topic_id=None,  # Will be set when associated with specific topics
            exam_name=exam_name,
            exam_date=exam_date,
            exam_weight=exam_weight_float
        )
        
        # Instead, we'll create a record in a separate table or handle differently
        # For now, let's create it without topic_id and handle it in the topic creation
        # Actually, let's create it with the first topic's ID if any exist, or skip for now
        
        # Better approach: Create exam dates associated with existing topics if requested
        selected_topics = request.form.getlist('topic_ids')
        
        if not selected_topics:
            # Create a standalone exam date entry (we'll need to modify schema for this)
            # For now, let's flash an error
            flash('Please select at least one topic to associate this exam date with', 'error')
            return redirect(url_for('manage'))
        
        created_associations = []
        for topic_id in selected_topics:
            if topic_id.strip():
                try:
                    topic_id_int = int(topic_id.strip())
                    topic = Topic.query.get(topic_id_int)
                    
                    if topic:
                        # Check if this topic already has this exam
                        existing_association = TopicExamDate.query.filter_by(
                            topic_id=topic_id_int,
                            exam_name=exam_name,
                            exam_date=exam_date
                        ).first()
                        
                        if not existing_association:
                            topic_exam_date = TopicExamDate(
                                topic_id=topic_id_int,
                                exam_name=exam_name,
                                exam_date=exam_date,
                                exam_weight=exam_weight_float
                            )
                            db.session.add(topic_exam_date)
                            created_associations.append(topic.name)
                
                except ValueError:
                    continue
        
        if created_associations:
            db.session.commit()
            flash(f'Exam "{exam_name}" on {exam_date} created and associated with topics: {", ".join(created_associations)}', 'success')
        else:
            flash('No valid topics selected for exam date association', 'error')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error creating exam date: {str(e)}', 'error')
    
    return redirect(url_for('manage'))

@app.route('/get_topics_json')
def get_topics_json():
    """API endpoint to get topics as JSON for dynamic loading"""
    topics = Topic.query.order_by(Topic.name).all()
    topics_data = [{'id': topic.id, 'name': topic.name, 'subject': topic.subject} for topic in topics]
    return jsonify(topics_data)

@app.route('/exam/<exam_id>')
def exam_progress(exam_id):
    states = db.session.execute("SELECT state FROM topics t JOIN topic_exam_dates ed on t.id=ed.topic_id WHERE ed.id='{{exam_id}}'")
    print(states.fetchone())
    return 'None'


if __name__ == '__main__':
    
    app.run(debug=True)