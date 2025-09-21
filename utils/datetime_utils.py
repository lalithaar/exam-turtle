import pytz
from datetime import datetime,date

ist = pytz.timezone('Asia/Kolkata')


def ensure_timezone_aware(dt, target_timezone=ist):
    """Ensure datetime object is timezone-aware and in IST"""
    if dt is None:
        return None
    
    if isinstance(dt, str):
        # If it's a string, parse it first
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
    if dt.tzinfo is not None:
        # Already timezone-aware, convert to IST
        return dt.astimezone(target_timezone)
    else:
        # Timezone-naive, assume it's in target timezone
        return target_timezone.localize(dt)

def now_ist():
    """Get current datetime in IST"""
    return datetime.now(ist)

def get_today():
    """Get today's date"""
    return date.today()
