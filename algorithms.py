from models import TopicExamDate,Topic
from app import db

def calculate_exam_progress(exam_id):
    exam_info = TopicExamDate.query.filter_by(id=exam_id).join(Topic).all()
    for i in range(len(exam_info)):
        print(
        exam_info[i].to_dict()
    )    
    return exam_info
