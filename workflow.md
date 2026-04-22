# Privacy-First Digital Twin Architecture Workflow

## Phase 1: The Silent Watchdog & Lifelong Learning
When you start the system (`telemetry_tracker.py`), a lightweight background process begins monitoring your activity. 

*   **Privacy First**: It never logs *what* you type (no keylogging). It only counts the *number of key presses* and the *pixel distance* your mouse travels inside a 10-second interval.
*   **The Baseline Engine**: As long as you are typing normally, it continuously saves these stats to a local CSV file. It calculates a rolling average of your "normal speed." If you become a faster typer over time, the system naturally learns and adapts to your new speed (handling "Concept Drift").

## Phase 2: The Suspicion Trigger
The Watchdog constantly compares your live 10-second interval stats against your established baseline. It gets suspicious if:
1.  **You go completely IDLE**: 0 keys and 0 mouse movement for 10 seconds.
2.  **You exhibit suspicious "LOW ACTIVITY"**: You type less than 5 keys and barely move the mouse.
3.  **Your Risk Score Spikes**: You are still typing, but your typing speed drastically plummets to less than 50% of your established baseline (and your mouse movement is minimal).

If any of these conditions are met, the Watchdog halts its assumptions. It realizes you *might* be asleep, but you also *might* just be reading an article. So, it triggers **The Cascade**.

## Phase 3: The Final Judge (Vision Core)
To eliminate false positives (annoying popups when you are just reading), the Watchdog silently boots up the AI Vision Core (`live_predictor.py`).

*   The webcam opens and uses MediaPipe to map your face.
*   It measures your **Eye Aspect Ratio (EAR)** for heavy blinks, **Mouth Aspect Ratio (MAR)** for yawning, and **Head Pitch** to see if your chin is slumping to your chest.
*   It feeds these coordinates into your pre-trained Support Vector Machine (SVM) brain, using rolling averages to ensure a single odd blink isn't counted as sleep.

## Phase 4: The Decision Fork
Once the Vision Core gets a lock on your face, one of two things happens:

*   **Scenario A (You are just reading)**: The AI classifies your face as `AWAKE` for about 1.5 seconds. It realizes the physical tracker made a mistake. The Vision Core instantly kills the camera process and returns a normal exit code to the Watchdog. The Watchdog resets its timers and resumes monitoring you. **No annoying popups are shown.**
*   **Scenario B (You are actually falling asleep)**: The AI classifies your face as `TIRED` for 30 consecutive frames (~1 second). The Vision Core immediately shuts down the camera to save power and sends a critical **Error Code 80** back to the Watchdog.

## Phase 5: The Cyber Intervention & Cooldown
The Watchdog receives Error Code 80 from the Vision Core. It now has double-verified proof that you are physically exhausted.

1.  **Deploy UI**: It launches `alert_ui.py`—a sleek, dark glassmorphism warning panel that floats above all your applications, commanding you to "Acknowledge & Rest."
2.  **Snooze Mode**: Knowing you need a break, the Watchdog honors a **5-Minute Cooldown/Snooze timer**. During this time, it continues recording your typing speed if you resume work, but it absolutely refuses to trigger the camera or throw another popup at you. 

Once the 5 minutes pass, the system smoothly returns to Phase 1, resuming its silent, protective watch over your digital workflow.
