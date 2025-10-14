import pytz
from datetime import datetime, date, timedelta

ist = pytz.timezone("Asia/Kolkata")


def ensure_timezone_aware(dt, target_timezone=ist):
    """Ensure datetime object is timezone-aware and in IST"""
    if dt is None:
        return None

    if isinstance(dt, str):
        # If it's a string, parse it first
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))

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


def is_due_today(due_datetime, reference_datetime=None):
    """Check if a due datetime falls on today's calendar date"""
    if reference_datetime is None:
        reference_datetime = now_ist()

    due_aware = ensure_timezone_aware(due_datetime)
    ref_aware = ensure_timezone_aware(reference_datetime)

    return due_aware.date() == ref_aware.date()


def is_due_tomorrow(due_datetime, reference_datetime=None):
    """Check if a due datetime falls on tomorrow's calendar date"""
    if reference_datetime is None:
        reference_datetime = now_ist()

    due_aware = ensure_timezone_aware(due_datetime)
    ref_aware = ensure_timezone_aware(reference_datetime)
    tomorrow = ref_aware.date() + timedelta(days=1)

    return due_aware.date() == tomorrow


def format_relative_timing(target_datetime, reference_datetime=None):
    """Format datetime relative to reference (or now) with proper day boundaries"""
    if reference_datetime is None:
        reference_datetime = now_ist()

    target_aware = ensure_timezone_aware(target_datetime)
    ref_aware = ensure_timezone_aware(reference_datetime)
    delta = target_aware - ref_aware

    # Check calendar date relationships
    target_date = target_aware.date()
    ref_date = ref_aware.date()

    if delta.total_seconds() < 0:
        # Overdue
        abs_delta = ref_aware - target_aware
        if abs_delta.days == 0:
            hours = abs_delta.total_seconds() // 3600
            return f"overdue by {int(hours)} hours"
        else:
            return (
                f"overdue by {abs_delta.days} day{'s' if abs_delta.days != 1 else ''}"
            )

    elif target_date == ref_date:
        # Due today
        hours = delta.total_seconds() // 3600
        if hours < 1:
            minutes = delta.total_seconds() // 60
            return f"due in {int(minutes)} minutes"
        else:
            return f"due in {int(hours)} hours (today)"

    elif target_date == ref_date + timedelta(days=1):
        # Due tomorrow
        hours = delta.total_seconds() // 3600
        return f"due in {int(hours)} hours (tomorrow)"

    elif delta.days <= 7:
        # Due this week
        day_name = target_aware.strftime("%A")
        return f"due {day_name} ({delta.days} days)"

    else:
        # Due later
        return f"due in {delta.days} days"

def get_this_week_s_monday()-> date:
    """returns the datetime object of this week's monday - week starting"""
    monday = get_today()-timedelta(get_today().weekday())
    return monday