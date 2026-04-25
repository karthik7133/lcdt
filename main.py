import subprocess
import sys
import os
import time

def run_cyber_watchdog():
    print("--- CYBER WATCHDOG INTEGRATED START ---")
    
    # 1. Start the Dashboard API in the background
    print("[SYSTEM] Starting Dashboard API (Backend)...")
    api_process = subprocess.Popen([sys.executable, 'api/dashboard_api.py'])
    
    # Small delay to let the API boot up
    time.sleep(2)
    
    print("[SYSTEM] Starting Telemetry Watchdog (Sensors)...")
    try:
        # 2. Run the Telemetry Tracker in the foreground
        # This keeps the terminal active for user feedback
        subprocess.run([sys.executable, '-u', 'sensors/telemetry_tracker.py'])
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping all components...")
    finally:
        # Cleanup: Ensure the API process is killed when exiting
        api_process.terminate()
        print("[SYSTEM] Dashboard API stopped.")

if __name__ == "__main__":
    run_cyber_watchdog()
