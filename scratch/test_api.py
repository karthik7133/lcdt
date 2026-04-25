"""
Tests every L6 policy button against the live API.
Run: python scratch/test_api.py
"""
import json, sys
import urllib.request

API = "http://localhost:5000/api/simulation"
POLICIES = [
    "baseline",
    "security_training",
    "reduce_notifications",
    "improve_sleep",
    "reduce_meetings",
    "pause_work",
    "adversarial_drill",
    "increased_workload",
]

results = {}
for policy in POLICIES:
    body = json.dumps({"policy": policy}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        if isinstance(data, list) and len(data) > 0:
            day1 = data[0]["Mean_Risk"]
            day30 = data[-1]["Mean_Risk"]
            results[policy] = (day1, day30)
            print(f"  {policy:25s}  Day1={day1:6.2f}%  Day30={day30:6.2f}%  OK")
        else:
            print(f"  {policy:25s}  ERROR: {data}")
    except Exception as e:
        print(f"  {policy:25s}  FAILED: {e}")

print()
print("--- Uniqueness check ---")
day30s = [v[1] for v in results.values()]
if len(set(day30s)) == len(day30s):
    print("PASS: All policies return DIFFERENT Day-30 Mean Risk values")
else:
    print("FAIL: Some policies return the SAME Day-30 value — check POLICIES dict!")
    for p, (d1, d30) in results.items():
        print(f"  {p}: {d30}")
