import time
import datetime
import math
import subprocess
import threading
import sys
import os
import pygetwindow as gw
from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
from pynput import keyboard, mouse
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import pythoncom

# Allow importing from the 'core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import baseline_engine
import context_api
import state_inference
import risk_forecaster

# --- INITIALIZE ENGINES ---
layer3_engine = state_inference.LatentStateInference()
layer4_engine = risk_forecaster.CyberRiskForecaster()

LATENT_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'latent_states.csv'))
DATA_DIR = os.path.dirname(LATENT_LOG)
LOCK_FILE = os.path.join(DATA_DIR, 'tracker.lock')

# --- SINGLETON CHECK ---
if os.path.exists(LOCK_FILE):
    # Check if the process is actually running (simple PID check)
    try:
        with open(LOCK_FILE, 'r') as f:
            old_pid = int(f.read().strip())
        import psutil
        if psutil.pid_exists(old_pid):
            print(f"[FATAL] Another instance of Telemetry Tracker is already running (PID: {old_pid}).")
            sys.exit(1)
    except:
        pass # If file is empty or corrupted, we'll overwrite it

# Create lock file with current PID
with open(LOCK_FILE, 'w') as f:
    f.write(str(os.getpid()))

print("--- TELEMETRY WATCHDOG V5 (CASCADE & COOLDOWN) ---")
print("Tracking physical input... (Press Ctrl+C in terminal to stop)")

# --- VARIABLES TO STORE DATA ---
key_count = 0
mouse_distance = 0.0
last_mouse_pos = None
last_activity_time = time.time()
camera_process = None 
cooldown_until = 0 # Keeps track of the Snooze Mode 
vision_fatigue_flag = False
good_password_flag = False

# --- NEW COGNITIVE BIOMARKERS ---
correction_count = 0
last_window_title = ""
window_change_count = 0
analytics_ticks = 0 # To track minutes (6 ticks = 1 minute)
session_start_time = time.time()
notification_count = 0
hourly_ticks = 0 # 360 ticks = 1 hour (10s intervals)
last_pw_mgr_time = 0
ctrl_pressed = False

# --- 0. NOTIFICATION LISTENER (AUDIO PEAK DETECTION) ---
def notification_audio_monitor():
    global notification_count
    pythoncom.CoInitialize() # Required for COM in threads
    
    last_peak_time = 0
    COOLDOWN = 1.5

    try:
        # Get the DEFAULT audio endpoint (captures ALL audio including system sounds)
        devices = AudioUtilities.GetSpeakers()
        interface = devices._dev.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
        meter = cast(interface, POINTER(IAudioMeterInformation))
    except Exception as e:
        print(f"  [ERROR] Failed to initialize audio meter: {e}")
        return

    while True:
        try:
            peak = meter.GetPeakValue()

            if peak > 0.05:
                current_time = time.time()
                if current_time - last_peak_time > COOLDOWN:
                    notification_count += 1
                    last_peak_time = current_time
                    print(f"  [DEBUG] Audio peak detected! Peak: {peak:.3f} | Total: {notification_count}")
                # Optional: print(f"  [DEBUG] Peak {peak:.3f} ignored (cooldown)")
            
        except Exception as e:
            # Re-initialize if the device was lost
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices._dev.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
                meter = cast(interface, POINTER(IAudioMeterInformation))
            except:
                pass
            time.sleep(1)

        time.sleep(0.05)  # Poll faster: 50ms
notif_thread = threading.Thread(target=notification_audio_monitor, daemon=True)
notif_thread.start()

# --- 1. KEYBOARD TRACKER ---
def on_press(key):
    global key_count, last_activity_time, correction_count, ctrl_pressed
    key_count += 1
    last_activity_time = time.time() 
    
    # Track Ctrl for Paste detection
    if key in [keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
        ctrl_pressed = True
        
    # Track "Correction" keys (Backspaces and Deletes)
    if key == keyboard.Key.backspace or key == keyboard.Key.delete:
        correction_count += 1
        
    # Detect Ctrl+V (Paste)
    try:
        if ctrl_pressed and hasattr(key, 'char') and key.char == 'v':
            # Check if paste happened shortly after a password manager was active
            if time.time() - last_pw_mgr_time < 30: # 30 second window
                print("[BEHAVIOUR] Good Password Habit: Paste detected after Password Manager usage.")
                good_password_flag = True
                with open(os.path.join(DATA_DIR, 'digital_behaviour.csv'), 'a', encoding='utf-8') as f:
                     f.write(f"{time.time()},GOOD_PASSWORD_HABIT,PASTE_FROM_MANAGER\n")
    except:
        pass

def on_release(key):
    global ctrl_pressed
    if key in [keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
        ctrl_pressed = False

# --- 2. MOUSE TRACKER ---
def on_move(x, y):
    global mouse_distance, last_mouse_pos, last_activity_time
    if last_mouse_pos is not None:
        dx = x - last_mouse_pos[0]
        dy = y - last_mouse_pos[1]
        mouse_distance += math.sqrt(dx**2 + dy**2)
    last_mouse_pos = (x, y)
    last_activity_time = time.time() 

# Start the listeners in the background
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_move=on_move)
keyboard_listener.start()
mouse_listener.start()

# --- 3. THE ANALYTICS LOOP ---
INTERVAL = 10 

try:
    while True:
        # We now use a 1-second sub-loop to catch rapid window switching
        for _ in range(INTERVAL):
            time.sleep(1)
            
            # --- FAST POLLING: TASK SWITCHING ---
            try:
                active_window = gw.getActiveWindowTitle()
                if active_window:
                    # Detect Password Managers
                    PASS_MANAGERS = ["Bitwarden", "1Password", "KeePass", "LastPass", "Dashlane"]
                    if any(pm in active_window for pm in PASS_MANAGERS):
                        last_pw_mgr_time = time.time()
                    
                    if active_window != last_window_title:
                        window_change_count += 1
                        last_window_title = active_window
            except:
                pass
        
        # --- THE REST OF THE 10-SECOND ANALYTICS ---
        
        # --- CAMERA PROCESS MANAGEMENT ---
        if camera_process is not None:
            if camera_process.poll() is None:
                is_active_now = (key_count >= 5 or mouse_distance >= 100)
                if is_active_now:
                    print("\n[Watchdog] User activity detected! Forcing Vision Core to close.")
                    camera_process.terminate()
                    camera_process = None
                    last_activity_time = time.time() 
                else:
                    print("[Watchdog] Camera is active. Monitoring for activity...")
                
                key_count = 0
                mouse_distance = 0.0
                continue
            else:
                ret_code = camera_process.returncode
                if ret_code == 80:
                    print("\n[Watchdog] AI Verified Fatigue! Deploying UI and entering 5-minute Snooze.")
                    subprocess.Popen(['python', 'ui/alert_ui.py'], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                    cooldown_until = time.time() + (5 * 60) # 5 Minute Snooze
                    
                    # Vision fatigue is now a temporary biomarker only
                    vision_fatigue_flag = True
                    
                    # Log intervention for Dashboard
                    with open(os.path.join(DATA_DIR, 'interventions.log'), 'a') as f:
                        f.write(f"Intervention at {time.time()}\n")
                else:
                    print("[Watchdog] Vision Core closed normally (Awake). Resetting activity clock.")
                last_activity_time = time.time() 
                camera_process = None

        # --- TELEMETRY ANALYTICS ---
        time_since_last_action = time.time() - last_activity_time
        baseline = baseline_engine.get_historical_baseline()
        base_keys = baseline['avg_keys']
        
        # Base risk includes system hygiene (Updates)
        update_risk = baseline_engine.get_system_update_risk()
        risk_score = 10 + update_risk 
        is_high_risk = False

        if base_keys > 5:
            performance_ratio = key_count / base_keys
            # Drop in typing MUST be accompanied by low mouse activity
            if performance_ratio < 0.5 and mouse_distance < 100: 
                risk_score = 80
                is_high_risk = True
            elif performance_ratio < 0.75:
                risk_score = 40

        # Save active baseline data if there is any physical activity
        if key_count > 0 or mouse_distance > 0:
             baseline_engine.save_active_stats(key_count, mouse_distance)

        # --- TRIGGER CONDITIONS ---
        # Strictest possible trigger: 60s of absolute zero activity
        # Removed is_high_risk bypass so camera only opens on physical stillness
        is_long_idle = time_since_last_action >= 60.0
        is_absolute_stillness = (key_count == 0 and mouse_distance == 0)
        
        if time.time() < cooldown_until:
             minutes_left = max(1, int((cooldown_until - time.time()) / 60))
             print(f"\n[STATE: COOLDOWN] {minutes_left} minutes until next AI check. Stats: {key_count} Keys")
        elif is_long_idle and is_absolute_stillness:
            print(f"\n[STATE: IDLE] -> Absolute stillness for 60s. Booting Vision Core.")
            camera_process = subprocess.Popen(['python', 'live_predictor.py'], cwd=os.path.join(os.path.dirname(__file__), 'fatigue_model_project'))
        else:
            print(f"\n[STATE: ACTIVE]")
            print(f"Current Stats: {key_count} Keys | {int(mouse_distance)}px Mouse")
            print(f"Learned Baseline: {int(base_keys)} Keys | Risk Score: {risk_score}%")

        analytics_ticks += 1
        
        # Performance/Cognitive Friction Logic
        if key_count > 0:
            error_ratio = correction_count / key_count
            if error_ratio > 0.15: # 15% correction rate is high
                print(f"[Biomarker] High Typing Friction: {int(error_ratio*100)}% corrections")
        
        # Check Task Switching every minute (6 intervals of 10s)
        if analytics_ticks >= 6:
            if window_change_count > 5:
                print(f"[Biomarker] High Task Switching Detected! ({window_change_count} changes/min)")
                with open(os.path.join(DATA_DIR, 'telemetry_events.csv'), 'a', encoding='utf-8') as f:
                    f.write(f"{time.time()},HIGH_TASK_SWITCHING,{window_change_count}\n")
            
            # Reset minute-based counters
            window_change_count = 0
            analytics_ticks = 0
        
        # Save live sensors for the frontend test page
        import json
        try:
            with open(os.path.join(DATA_DIR, 'live_sensors.json'), 'w') as f:
                json.dump({
                    "task_switches": window_change_count,
                    "notifications": notification_count,
                    "correction_count": correction_count,
                    "key_count": key_count
                }, f)
        except Exception as e:
            pass
            
        # Log notifications EVERY interval (10s) for testing
        context_api.log_notifications(notification_count)
        notification_count = 0
        hourly_ticks = 0 # Not used for notifications anymore, but kept for structure

        # --- LAYER 2: STATE INFERENCE ---
        # 1. Read external signals from CSVs
        # CRITICAL: Only read events that happened in the LAST 30 seconds.
        # Reading tail(100) of all-time historical data caused old test events
        # (e.g. INSECURE_HTTP from April 21) to drain Ht every 10 seconds.
        import pandas as pd
        import re
        insecure_hits = 0
        webmail_hits = 0
        email_frequency = 0
        link_clicks = 0
        low_strength_passwords = 0
        unknown_senders = 0
        avg_response_time = 0.0
        phishing_clicked = "FALSE"
        scam_given = "FALSE"
        hour_of_day = datetime.datetime.now().hour # Default to local system hour

        # Time window: only count events from the last 30 seconds
        SIGNAL_WINDOW_SECS = 30
        now_ts = time.time()
        since_ts = now_ts - SIGNAL_WINDOW_SECS

        try:
            if os.path.exists(os.path.join(DATA_DIR, 'digital_behaviour.csv')):
                df_beh = pd.read_csv(os.path.join(DATA_DIR, 'digital_behaviour.csv'))
                # Normalize timestamp column to Unix float (handles both ISO and epoch formats)
                def to_unix(ts):
                    try:
                        f = float(ts)
                        return f  # Already a Unix epoch float
                    except (ValueError, TypeError):
                        try:
                            import datetime
                            return pd.Timestamp(ts).timestamp()
                        except:
                            return 0.0

                df_beh['ts_unix'] = df_beh['timestamp'].apply(to_unix)
                # Only events from the last 30 seconds
                df_beh = df_beh[df_beh['ts_unix'] >= since_ts]

                if not df_beh.empty:
                    insecure_hits = len(df_beh[df_beh['event'] == 'INSECURE_HTTP'])
                    webmail_hits = len(df_beh[df_beh['event'] == 'WEBMAIL_ACCESS'])
                    email_frequency = len(df_beh[df_beh['event'] == 'EMAIL_SENDERS_DETECTED'])
                    link_clicks = len(df_beh[df_beh['event'] == 'LINK_CLICK'])
                    low_strength_passwords = len(df_beh[df_beh['event'] == 'LOW_STRENGTH_PASSWORD'])

                    if 'details' in df_beh.columns:
                        response_times_df = df_beh[df_beh['event'] == 'RESPONSE_TIME']
                        if not response_times_df.empty:
                            total_time = 0
                            count = 0
                            for details in response_times_df['details']:
                                match = re.search(r'"response_time_ms":\s*(\d+)', str(details))
                                if match:
                                    total_time += int(match.group(1))
                                    count += 1
                            if count > 0:
                                avg_response_time = total_time / count

                        sender_df = df_beh[df_beh['event'] == 'EMAIL_SENDERS_DETECTED']
                        for details in sender_df['details']:
                            match = re.search(r'"count":\s*(\d+)', str(details))
                            if match:
                                unknown_senders += int(match.group(1))

                        # Extract hour of day from the latest event
                        latest_details = str(df_beh.iloc[-1]['details'])
                        match_hour = re.search(r'"hour_of_day":\s*(\d+)', latest_details)
                        if match_hour:
                            hour_of_day = int(match_hour.group(1))

        except Exception as e:
            pass  # If anything fails, signals stay at 0 (safe default)

        try:
            if os.path.exists(os.path.join(DATA_DIR, 'adversarial_failures.csv')):
                df_adv = pd.read_csv(os.path.join(DATA_DIR, 'adversarial_failures.csv'))
                # Only look at adversarial events from the last 5 minutes
                df_adv['ts_unix'] = pd.to_numeric(df_adv['timestamp'], errors='coerce')
                df_adv_recent = df_adv[df_adv['ts_unix'] >= (now_ts - 300)]
                if not df_adv_recent.empty:
                    if any("Phishing" in str(x) for x in df_adv_recent['attack_type']): phishing_clicked = "TRUE"
                    if any("Scam" in str(x) for x in df_adv_recent['attack_type']): scam_given = "TRUE"
        except:
            pass

        # FIX: Read SLEEP_DEFICIT from the LATEST entry in work_context.csv only.
        # Also enforces a 4-hour expiry — if the latest SLEEP_DEFICIT,TRUE was
        # written more than 4 hours ago (e.g. from a testing session earlier today),
        # it is treated as stale/expired and does NOT suppress Ct.
        sleep_deficit_active = 'FALSE'
        SLEEP_EXPIRY_SECS = 4 * 3600  # 4 hours
        try:
            if os.path.exists(os.path.join(DATA_DIR, 'work_context.csv')):
                df_ctx = pd.read_csv(os.path.join(DATA_DIR, 'work_context.csv'))
                sleep_rows = df_ctx[df_ctx['event_type'] == 'SLEEP_DEFICIT']
                if not sleep_rows.empty:
                    latest_sleep = sleep_rows.iloc[-1]
                    # Parse the timestamp of the latest sleep entry
                    try:
                        sleep_ts = pd.Timestamp(str(latest_sleep['timestamp'])).timestamp()
                        sleep_age_secs = now_ts - sleep_ts
                    except:
                        sleep_age_secs = 0  # If can't parse, assume recent
                    is_recent = sleep_age_secs <= SLEEP_EXPIRY_SECS
                    is_true = str(latest_sleep['value']).strip().upper() == 'TRUE'
                    sleep_deficit_active = 'TRUE' if (is_true and is_recent) else 'FALSE'
                    if is_true and not is_recent:
                        print(f"[INFO] SLEEP_DEFICIT entry is {int(sleep_age_secs/3600)}h old — treated as expired.")
        except:
            pass

        # 2. Pack the Signal Vector
        # FIX: task_switches uses a threshold of >4 before passing to signals.
        # VS Code changes its window title on EVERY file open, save, extension
        # event and git status update — so 1-4 switches per 10s is just IDE noise.
        # Only genuine multi-tasking (>4 switches) should count as cognitive demand.
        effective_task_switches = max(0, window_change_count - 4)  # subtract IDE noise floor
        signals = {
            'sleep_deficit': sleep_deficit_active,
            'vision_fatigue': 'TRUE' if (vision_fatigue_flag or time.time() < cooldown_until) else 'FALSE',
            'task_switches': effective_task_switches,        # IDE-noise-subtracted task switches
            'notification_count': notification_count,       # Audio notifications this interval
            'key_count': key_count,                         # Raw key count (gates error rate logic)
            'base_keys': base_keys,                         # Added baseline for performance detection
            'typing_error_rate': correction_count / max(1, key_count),
            'mouse_entropy': mouse_distance,                # Raw px moved (inference gates at >200px)
            'insecure_http_hits': insecure_hits,            # Only events in last 30s
            'webmail_hits': webmail_hits,                   # Only events in last 30s
            'email_frequency': email_frequency,
            'link_clicks': link_clicks,
            'low_strength_passwords': low_strength_passwords,
            'unknown_senders': unknown_senders,
            'avg_response_time': avg_response_time,
            'hour_of_day': hour_of_day, # Added from extension
            'os_update_delayed': 'TRUE' if update_risk > 0 else 'FALSE',
            'good_password_paste': 'TRUE' if good_password_flag else 'FALSE',
            'phishing_clicked': phishing_clicked,
            'scam_credentials_given': scam_given
        }

        # 3. Engage Inference Engine (Layer 3)
        z_t = layer3_engine.update_inference(signals)
        
        # 4. Calculate Predictive Risk (Layer 4)
        risk_pct = layer4_engine.calculate_live_risk(z_t)
        
        print(f"\n[Z_t INFERENCE] Capacity: {z_t['Capacity_Ct']} | Demand: {z_t['Demand_Dt']} | Reserve: {z_t['Reserve_Gap_CRGt']}")
        print(f"[Z_t HYGIENE] Habits: {z_t['Habits_Ht']} | Adversarial: {z_t['Adversarial_At']}")
        print(f"[P_Mt RISK] Mistake Probability: {risk_pct}%")

        # 5. Log Latent State for Dashboard
        if not os.path.exists(LATENT_LOG):
            with open(LATENT_LOG, 'w', encoding='utf-8') as f:
                f.write("timestamp,Ct,Dt,Ht,At,CRGt,risk_pct\n")
        
        with open(LATENT_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{time.time()},{z_t['Capacity_Ct']},{z_t['Demand_Dt']},{z_t['Habits_Ht']},{z_t['Adversarial_At']},{z_t['Reserve_Gap_CRGt']},{risk_pct}\n")

        # Reset per-interval flags
        vision_fatigue_flag = False
        good_password_flag = False

        key_count = 0
        mouse_distance = 0.0
        correction_count = 0 # Reset corrections per interval
        
except KeyboardInterrupt:
    print("\nWatchdog stopped.")
    if camera_process and camera_process.poll() is None:
        camera_process.terminate()
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    keyboard_listener.stop()
    mouse_listener.stop()
