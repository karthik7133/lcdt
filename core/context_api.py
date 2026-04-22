import os
import time
from datetime import datetime

CONTEXT_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'work_context.csv'))

def log_context_event(event_type, value):
    """Logs a context event (Sleep Deficit, Notification Count, etc.)"""
    if not os.path.exists(CONTEXT_LOG):
        with open(CONTEXT_LOG, 'w', encoding='utf-8') as f:
            f.write("timestamp,event_type,value\n")
            
    with open(CONTEXT_LOG, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{event_type},{value}\n")

def check_sleep_deficit(is_fatigued, session_start_time):
    """
    Checks if the user is fatigued within the first hour of usage.
    If so, logs a Sleep Deficit flag.
    """
    elapsed_time = time.time() - session_start_time
    if is_fatigued and elapsed_time < 3600: # 1 hour
        log_context_event("SLEEP_DEFICIT", "TRUE")
        print("[CONTEXT] Sleep Deficit detected (Fatigue in first hour of usage)!")
        return True
    return False

def log_notifications(count):
    """Logs the notification count for the current interval."""
    log_context_event("NOTIFICATION_COUNT", count)
    print(f"[CONTEXT] Logged {count} notifications for the past 10 seconds.")
