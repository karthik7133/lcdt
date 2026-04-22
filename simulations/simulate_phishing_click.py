import requests
import json

def simulate_click():
    url = "http://localhost:5000/api/adversarial_signal"
    payload = {
        "type": "Phishing Simulation",
        "status": "CLICKED_LINK",
        "details": "User clicked on a simulated malicious login page while fatigued."
    }
    
    print(f"Sending simulated attack signal to {url}...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Successfully logged adversarial failure!")
        else:
            print(f"Failed to log. Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}. Make sure dashboard_api.py is running.")

if __name__ == "__main__":
    simulate_click()
