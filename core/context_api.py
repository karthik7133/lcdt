"""
core/context_api.py
====================
Provides two things:
  1. GoogleContextEngine — reads Google Calendar to produce a
     workload_modifier for Layer 2 (one-time fetch at startup).
  2. log_notifications() — called by telemetry_tracker.py every 10s
     to persist audio-peak interrupt counts to work_context.csv.
"""

import os
import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ─────────────────────────────────────────────────────────────
# HELPER — Notification Logger (used by telemetry_tracker.py)
# ─────────────────────────────────────────────────────────────

def log_notifications(count: int):
    """
    Appends an audio-notification count record to work_context.csv.
    Called by telemetry_tracker.py every 10-second interval.
    """
    if count <= 0:
        return
    ts  = datetime.datetime.now().isoformat()
    row = f"{ts},NOTIFICATION_COUNT,{count}\n"
    path = os.path.join(DATA_DIR, 'work_context.csv')
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(row)
    except Exception as e:
        print(f"[context_api] Could not log notifications: {e}")


# ─────────────────────────────────────────────────────────────
# GOOGLE CALENDAR ENGINE — Layer 1 Work-Context Signal
# ─────────────────────────────────────────────────────────────

class GoogleContextEngine:
    """
    Authenticates with the Google Calendar API (OAuth 2.0) and
    fetches today's schedule to produce a workload_modifier [0.0–0.5]
    that seeds the Cognitive Demand (Dt) baseline in Layer 2.
    """

    def __init__(self):
        # Read-only calendar access
        self.SCOPES    = ['https://www.googleapis.com/auth/calendar.readonly']
        self.creds_file = os.path.join(os.path.dirname(__file__), 'credentials.json')
        self.token_file = os.path.join(os.path.dirname(__file__), 'token.json')

    def authenticate(self):
        """Handles OAuth 2.0 flow. Saves token for future sessions."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(
                self.token_file, self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow  = InstalledAppFlow.from_client_secrets_file(
                    self.creds_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'w') as f:
                f.write(creds.to_json())

        return creds

    def fetch_daily_workload(self) -> float:
        """
        Fetches today's Google Calendar events and maps meeting count
        to a Cognitive Demand modifier:
          0 meetings  → 0.0  (no added pressure)
          1-2         → 0.1  (light day)
          3-4         → 0.3  (moderate day)
          5+          → 0.5  (heavy day — high starting Dt)

        Returns the modifier as a float in [0.0, 0.5].
        Falls back to 0.0 gracefully if auth or network fails.
        """
        try:
            from googleapiclient.discovery import build

            creds   = self.authenticate()
            service = build('calendar', 'v3', credentials=creds)

            now = datetime.datetime.utcnow()
            t_min = now.replace(hour=0,  minute=0,  second=0,  microsecond=0).isoformat() + 'Z'
            t_max = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat() + 'Z'

            print("[LAYER 1] Connecting to Google Calendar API...")
            result = service.events().list(
                calendarId='primary',
                timeMin=t_min, timeMax=t_max,
                singleEvents=True, orderBy='startTime'
            ).execute()

            events = result.get('items', [])
            n      = len(events)
            print(f"[LAYER 1] {n} meetings found today.")
            for ev in events:
                start = ev['start'].get('dateTime', ev['start'].get('date'))
                print(f"   -> {start}: {ev.get('summary', 'Busy')}")

            if   n == 0:   return 0.0
            elif n <= 2:   return 0.1
            elif n <= 4:   return 0.3
            else:          return 0.5

        except Exception as e:
            print(f"[LAYER 1] Calendar API unavailable ({e}). Using 0.0 workload.")
            return 0.0


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine   = GoogleContextEngine()
    modifier = engine.fetch_daily_workload()
    print(f"\nWorkload Modifier for Layer 2 Dt baseline: +{modifier}")
