from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import pythoncom
import time

pythoncom.CoInitialize()

try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices._dev.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
    meter = cast(interface, POINTER(IAudioMeterInformation))
    print("Success! Peak is:", meter.GetPeakValue())
except Exception as e:
    print(f"Error: {e}")
