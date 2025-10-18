class PomodoroSession:
    def __init__(self,start_time,topic_id,end_time=None,review_id=None):
        self.start_time = start_time
        self.end_time = end_time
        self.topic = topic_id
        self.review_id = review_id
