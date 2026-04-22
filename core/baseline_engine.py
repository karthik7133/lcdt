import pandas as pd
import os
import subprocess
from datetime import datetime

CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry_history.csv'))

def save_active_stats(keys_pressed, mouse_distance):
    """Saves the stats to a CSV to build a lifelong learning baseline."""
    # Create file with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['timestamp', 'keys_pressed', 'mouse_distance'])
        df.to_csv(CSV_FILE, index=False)

    # Append the new active data
    new_data = pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'keys_pressed': keys_pressed,
        'mouse_distance': mouse_distance
    }])
    new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)

def get_historical_baseline():
    """Calculates the historical average speed of the user."""
    if not os.path.exists(CSV_FILE):
        return {"avg_keys": 0, "avg_mouse": 0} # No history yet

    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) == 0:
            return {"avg_keys": 0, "avg_mouse": 0}
        
        # Calculate the historical average
        avg_keys = df['keys_pressed'].mean()
        avg_mouse = df['mouse_distance'].mean()
        
        return {"avg_keys": avg_keys, "avg_mouse": avg_mouse}
    except Exception as e:
        return {"avg_keys": 0, "avg_mouse": 0}

def get_system_update_risk():
    """Checks if the last system update is older than 30 days."""
    try:
        # Run wmic to get update dates
        output = subprocess.check_output("wmic qfe get InstalledOn", shell=True).decode()
        dates = []
        for line in output.splitlines():
            line = line.strip()
            if not line or "InstalledOn" in line:
                continue
            try:
                # WMIC usually returns M/D/YYYY
                dt = datetime.strptime(line, "%m/%d/%Y")
                dates.append(dt)
            except:
                try:
                    # Sometimes it's YYYYMMDD
                    dt = datetime.strptime(line, "%Y%m%d")
                    dates.append(dt)
                except:
                    continue
        
        if not dates:
            return 0
            
        last_update = max(dates)
        days_since_update = (datetime.now() - last_update).days
        
        if days_since_update > 30:
            print(f"[ENGINE] Security Risk: Last update was {days_since_update} days ago!")
            return 20 # Risk points
        return 0
    except Exception as e:
        print(f"Error checking updates: {e}")
        return 0
