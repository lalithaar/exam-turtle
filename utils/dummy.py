from db import db
from app import app
from models import Topic, TopicReviewLog, TopicExamDate, TopicScheduler, State, Rating,ensure_timezone_aware
from datetime import datetime, timezone, timedelta, date
import random
import math

def create_db():
    """Create all database tables"""
    with app.app_context():
        db.create_all()
    print("✅ Database tables created successfully!")

def populate_dummy_data():
    """Populate database with realistic dummy data"""
    
    # Clear existing data
    db.session.query(TopicReviewLog).delete()
    db.session.query(TopicExamDate).delete()
    db.session.query(Topic).delete()
    db.session.commit()
    
    # Create scheduler
    scheduler = TopicScheduler()
    
    # Define subjects and their topics
    subjects_data = {
        "Mathematics": [
            "Calculus - Limits and Continuity",
            "Calculus - Integration by Parts", 
            "Linear Algebra - Matrix Operations",
            "Linear Algebra - Eigenvalues and Eigenvectors",
            "Differential Equations - First Order",
            "Probability Theory - Bayes Theorem",
            "Statistics - Hypothesis Testing",
            "Complex Analysis - Cauchy's Theorem"
        ],
        "Physics": [
            "Thermodynamics - First Law",
            "Thermodynamics - Entropy and Second Law",
            "Quantum Mechanics - Schrödinger Equation",
            "Quantum Mechanics - Wave-Particle Duality", 
            "Electromagnetism - Maxwell's Equations",
            "Electromagnetism - Electromagnetic Induction",
            "Mechanics - Lagrangian Formulation",
            "Optics - Interference and Diffraction"
        ],
        "Chemistry": [
            "Organic Chemistry - Reaction Mechanisms",
            "Organic Chemistry - Stereochemistry",
            "Physical Chemistry - Chemical Kinetics",
            "Physical Chemistry - Thermochemistry",
            "Inorganic Chemistry - Coordination Compounds",
            "Inorganic Chemistry - Crystal Field Theory",
            "Analytical Chemistry - Spectroscopy",
            "Biochemistry - Enzyme Kinetics"
        ],
        "Computer Science": [
            "Data Structures - Binary Trees",
            "Data Structures - Hash Tables",
            "Algorithms - Dynamic Programming",
            "Algorithms - Graph Algorithms",
            "Database Systems - SQL Joins",
            "Database Systems - Indexing",
            "Operating Systems - Process Scheduling",
            "Networks - TCP/IP Protocol"
        ]
    }
    
    # Create topics
    topics = []
    topic_id = 1
    
    for subject, topic_names in subjects_data.items():
        for topic_name in topic_names:
            # Create topic with varied initial states
            initial_state = random.choice([State.Learning, State.Review, State.Relearning])
            
            topic = Topic(
                name=topic_name,
                subject=subject,
                state=initial_state,
                step=0 if initial_state == State.Learning else None
            )
            
            # Set creation time (topics created over last 60 days)
            days_ago = random.randint(1, 60)
            topic.created_at = ensure_timezone_aware(datetime.now() - timedelta(days=days_ago))

            # If not a new topic, give it some FSRS parameters
            if initial_state != State.Learning or random.random() < 0.7:
                # Give it some initial stability and difficulty based on simulated history
                num_reviews = random.randint(1, 8)
                
                # Start with initial values
                current_stability = scheduler._initial_stability(Rating.Good)
                current_difficulty = scheduler._initial_difficulty(Rating.Good)
                last_review_date = topic.created_at
                
                # Simulate review history
                for review_num in range(num_reviews):
                    # Time between reviews (1-14 days)
                    days_between = random.randint(1, 14)
                    review_date = last_review_date + timedelta(days=days_between)
                    
                    # Generate realistic rating based on retrievability
                    if review_num == 0:
                        rating = random.choices(
                            [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
                            weights=[0.1, 0.2, 0.5, 0.2]
                        )[0]
                    else:
                        # Calculate retrievability to influence rating
                        elapsed_days = (review_date - last_review_date).days
                        retrievability = (1 + (scheduler._FACTOR * elapsed_days) / current_stability) ** scheduler._DECAY
                        
                        if retrievability < 0.5:
                            rating = random.choices([Rating.Again, Rating.Hard], weights=[0.7, 0.3])[0]
                        elif retrievability < 0.8:
                            rating = random.choices([Rating.Hard, Rating.Good], weights=[0.3, 0.7])[0]
                        else:
                            rating = random.choices([Rating.Good, Rating.Easy], weights=[0.6, 0.4])[0]
                    
                    # Update FSRS parameters
                    if rating == Rating.Again:
                        current_stability = scheduler._next_forget_stability(
                            current_difficulty, current_stability, retrievability if review_num > 0 else 0.5
                        )
                    else:
                        current_stability = scheduler._next_recall_stability(
                            current_difficulty, current_stability, 
                            retrievability if review_num > 0 else 0.8, rating
                        )
                    
                    current_difficulty = scheduler._next_difficulty(current_difficulty, rating)
                    last_review_date = review_date
                
                # Set final FSRS state
                topic.stability = current_stability
                topic.difficulty = current_difficulty
                topic.last_review = last_review_date
                
                # Set due date based on current state
                if topic.state == State.Review:
                    next_interval_days = scheduler._next_interval(current_stability)
                    # Make some topics overdue, some due soon, some due later
                    fuzz_factor = random.uniform(0.5, 1.5)
                    if random.random() < 0.3:  # 30% overdue
                        fuzz_factor = random.uniform(0.3, 0.8)
                    elif random.random() < 0.5:  # 20% due very soon
                        fuzz_factor = random.uniform(0.8, 1.1)
                    
                    topic.due = last_review_date + timedelta(days=int(next_interval_days * fuzz_factor))
                else:
                    # Learning/Relearning topics due sooner
                    topic.due = last_review_date + timedelta(hours=random.randint(1, 48))
            
            db.session.add(topic)
            topics.append(topic)
            topic_id += 1
    
    db.session.commit()
    
    # Create exam dates for topics
    exam_types = [
        ("Mid-term Exam", 25, 1.5),      # 25 days from now, weight 1.5
        ("Final Exam", 45, 2.0),         # 45 days from now, weight 2.0
        ("Quiz 1", 8, 0.8),              # 8 days from now, weight 0.8
        ("Quiz 2", 18, 0.8),             # 18 days from now, weight 0.8
        ("Assignment Due", 12, 1.0),     # 12 days from now, weight 1.0
        ("Lab Exam", 35, 1.3),           # 35 days from now, weight 1.3
    ]

    current_date = ensure_timezone_aware(datetime.now()).date()

    for topic in topics:
        # Each topic gets 1-3 exam dates
        num_exams = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        selected_exams = random.sample(exam_types, num_exams)
        
        for exam_name, days_ahead, weight in selected_exams:
            # Add some randomness to exam dates (±3 days)
            days_variance = random.randint(-3, 3)
            exam_date = current_date + timedelta(days=days_ahead + days_variance)
            
            exam = TopicExamDate(
                topic_id=topic.id,
                exam_name=f"{topic.subject} - {exam_name}",
                exam_date=exam_date,
                exam_weight=weight
            )
            db.session.add(exam)
    
    db.session.commit()
    
    # Create review logs for topics that have been reviewed
    for topic in topics:
        if topic.last_review and topic.stability:
            # Create 2-6 review logs per topic
            num_logs = random.randint(2, 6)
            
            # Start from topic creation and work forward
            current_date = topic.created_at
            current_stability = scheduler._initial_stability(Rating.Good)
            current_difficulty = scheduler._initial_difficulty(Rating.Good)
            
            for log_num in range(num_logs):
                # Time between reviews
                if log_num == 0:
                    days_between = random.randint(1, 3)  # First review soon after creation
                else:
                    days_between = random.randint(1, int(current_stability * 0.8) + 1)
                
                review_date = current_date + timedelta(days=days_between)
                
                # Don't create logs beyond the topic's last_review date
                if review_date > topic.last_review:
                    break
                
                # Generate rating based on retrievability
                if log_num == 0:
                    rating = random.choices(
                        [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
                        weights=[0.1, 0.2, 0.5, 0.2]
                    )[0]
                else:
                    elapsed_days = (review_date - current_date).days
                    retrievability = (1 + (scheduler._FACTOR * elapsed_days) / current_stability) ** scheduler._DECAY
                    
                    if retrievability < 0.5:
                        rating = Rating.Again
                    elif retrievability < 0.7:
                        rating = random.choice([Rating.Again, Rating.Hard])
                    elif retrievability < 0.9:
                        rating = random.choice([Rating.Hard, Rating.Good])
                    else:
                        rating = random.choice([Rating.Good, Rating.Easy])
                
                # Convert rating to retention percentage
                retention_map = {
                    Rating.Again: random.randint(20, 55),
                    Rating.Hard: random.randint(60, 74),
                    Rating.Good: random.randint(75, 89),  
                    Rating.Easy: random.randint(90, 100)
                }
                retention_percentage = retention_map[rating]
                
                # Review duration (30 seconds to 20 minutes)
                review_duration = random.randint(30, 1200)
                
                # Store previous state
                stability_before = current_stability
                difficulty_before = current_difficulty
                
                # Update FSRS parameters
                elapsed_days = max(0, (review_date - current_date).days)
                if elapsed_days > 0:
                    retrievability = (1 + (scheduler._FACTOR * elapsed_days) / current_stability) ** scheduler._DECAY
                else:
                    retrievability = 0.9
                
                if rating == Rating.Again:
                    current_stability = scheduler._next_forget_stability(
                        current_difficulty, current_stability, retrievability
                    )
                else:
                    current_stability = scheduler._next_recall_stability(
                        current_difficulty, current_stability, retrievability, rating
                    )
                
                current_difficulty = scheduler._next_difficulty(current_difficulty, rating)
                
                # Create review log
                review_log = TopicReviewLog(
                    topic_id=topic.id,
                    rating=rating,
                    retention_percentage=retention_percentage,
                    review_datetime=review_date,
                    review_duration=review_duration,
                    stability_before=stability_before,
                    difficulty_before=difficulty_before,
                    stability_after=current_stability,
                    difficulty_after=current_difficulty
                )
                db.session.add(review_log)
                current_date = review_date
    
    db.session.commit()
    
    # Print summary statistics
    total_topics = Topic.query.count()
    total_logs = TopicReviewLog.query.count()
    total_exams = TopicExamDate.query.count()

    overdue_topics = Topic.query.filter(Topic.due < ensure_timezone_aware(datetime.now())).count()
    due_today = Topic.query.filter(
        Topic.due >= ensure_timezone_aware(datetime.now()).date(),
        Topic.due < ensure_timezone_aware(datetime.now()).date() + timedelta(days=1)
    ).count()
    
    print(f"\n📊 Dummy Data Summary:")
    print(f"   • Topics created: {total_topics}")
    print(f"   • Review logs: {total_logs}")
    print(f"   • Exam dates: {total_exams}")
    print(f"   • Overdue topics: {overdue_topics}")
    print(f"   • Due today: {due_today}")
    
    # Show subject breakdown
    subjects = db.session.query(Topic.subject, db.func.count(Topic.id)).group_by(Topic.subject).all()
    print(f"\n📚 Topics by Subject:")
    for subject, count in subjects:
        print(f"   • {subject}: {count} topics")
    
    print(f"\n✅ Dummy data populated successfully!")
    return topics

def get_priority_topics_demo(limit=3):
    """Demo function to show priority topic selection"""
    scheduler = TopicScheduler()
    priority_topics = scheduler.get_priority_topics(limit=limit)
    
    print(f"\n🎯 Top {limit} Priority Topics for Review:")
    print("=" * 60)

    current_time = ensure_timezone_aware(datetime.now())

    for i, topic in enumerate(priority_topics, 1):
        retrievability = topic.get_current_retrievability(current_time)
        exam_urgency = topic.get_exam_urgency(current_time)
        priority_score = topic.calculate_priority_score(current_time)
        
        # Get next exam info
        upcoming_exams = [
            exam for exam in topic.exam_dates 
            if exam.exam_date >= current_time.date()
        ]
        next_exam = min(upcoming_exams, key=lambda x: x.exam_date) if upcoming_exams else None

        days_overdue = (current_time - ensure_timezone_aware(topic.due)).days if ensure_timezone_aware(topic.due) < current_time else 0

        print(f"{i}. {topic.name}")
        print(f"   Subject: {topic.subject}")
        print(f"   Current Retrievability: {retrievability:.1%}")
        print(f"   Forgetting Probability: {(1-retrievability):.1%}")
        print(f"   Exam Urgency: {exam_urgency:.3f}")
        print(f"   Priority Score: {priority_score:.3f}")
        print(f"   State: {State(topic.state).name}")
        print(f"   Difficulty: {topic.difficulty:.2f}/10" if topic.difficulty else "   Difficulty: New topic")
        print(f"   Stability: {topic.stability:.1f} days" if topic.stability else "   Stability: New topic")
        
        if days_overdue > 0:
            print(f"   ⚠️  OVERDUE by {days_overdue} days")
        elif topic.due.date() == current_time.date():
            print(f"   📅 Due TODAY")
        else:
            days_until_due = (topic.due.date() - current_time.date()).days
            print(f"   📅 Due in {days_until_due} days")
        
        if next_exam:
            days_to_exam = (next_exam.exam_date - current_time.date()).days
            print(f"   🎓 Next exam: {next_exam.exam_name} in {days_to_exam} days")
        
        print()

def simulate_review_session():
    """Simulate a review session with the top priority topics"""
    scheduler = TopicScheduler()
    priority_topics = scheduler.get_priority_topics(limit=3)
    
    print("🎯 Starting Review Session...")
    print("=" * 50)
    
    for topic in priority_topics:
        print(f"\nReviewing: {topic.name}")
        print(f"Subject: {topic.subject}")
        
        # Simulate user rating (weighted toward realistic performance)
        current_retrievability = topic.get_current_retrievability()
        
        if current_retrievability < 0.5:
            rating = random.choices([Rating.Again, Rating.Hard], weights=[0.8, 0.2])[0]
        elif current_retrievability < 0.8:
            rating = random.choices([Rating.Hard, Rating.Good], weights=[0.3, 0.7])[0]
        else:
            rating = random.choices([Rating.Good, Rating.Easy], weights=[0.7, 0.3])[0]
        
        # Simulate retention percentage and review duration
        retention_map = {
            Rating.Again: random.randint(30, 55),
            Rating.Hard: random.randint(60, 74),
            Rating.Good: random.randint(75, 89),
            Rating.Easy: random.randint(90, 100)
        }
        retention_percentage = retention_map[rating]
        review_duration = random.randint(60, 600)  # 1-10 minutes
        
        print(f"User rated: {Rating(rating).name} ({retention_percentage}% retention)")
        print(f"Review duration: {review_duration//60}m {review_duration%60}s")
        
        # Process the review
        updated_topic, review_log = scheduler.review_topic(
            topic, rating, retention_percentage=retention_percentage, 
            review_duration=review_duration
        )
        
        # Update database
        db.session.merge(updated_topic)
        db.session.add(review_log)
        
        print(f"Updated difficulty: {updated_topic.difficulty:.2f}/10")
        print(f"Updated stability: {updated_topic.stability:.1f} days")
        print(f"Next review due: {updated_topic.due.strftime('%Y-%m-%d %H:%M')}")
        print(f"New state: {State(updated_topic.state).name}")
    
    db.session.commit()
    print(f"\n✅ Review session completed! Database updated.")

if __name__ == "__main__":
    with app.app_context():
        print("🚀 Creating database and populating with dummy data...")
    
    # Create database
        create_db()
    
    # Populate with dummy data
        topics = populate_dummy_data()
    
    # Show priority topics demo
        get_priority_topics_demo(limit=5)
    
    # Run a simulated review session
        simulate_review_session()
    
    # Show updated priorities
        print("\n" + "="*60)
        print("📈 UPDATED PRIORITIES AFTER REVIEW SESSION:")
        get_priority_topics_demo(limit=5)