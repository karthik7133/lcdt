import time
from pycaw.pycaw import AudioUtilities, IAudioMeterInformation

print("--- Audio Session Monitor ---")
print("Playing sound now? I will list all processes with peak > 0.01")

try:
    while True:
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            try:
                meter = session._ctl.QueryInterface(IAudioMeterInformation)
                peak = meter.GetPeakValue()
                if peak > 0.01:
                    process_name = session.Process.name() if session.Process else "System/Unknown"
                    print(f"Process: {process_name} | Peak: {peak:.4f}")
            except Exception:
                pass
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopped.")
