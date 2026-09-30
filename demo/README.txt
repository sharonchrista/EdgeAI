SMART MOTOR EDGE AI CLASSROOM DEMO — LIVE DASHBOARD EDITION
===========================================================

WHAT IT DEMONSTRATES
--------------------
1. MQTT publisher          -> sensor.py
2. MQTT broker             -> Mosquitto on localhost:1883
3. MQTT subscriber         -> app.py
4. Telemetry               -> temperature + vibration messages
5. Lightweight database    -> SQLite file motor.db
6. Edge analytics          -> rolling-window RMS + thresholds
7. Predictive maintenance  -> NORMAL / WARNING / FAULT
8. MQTT alert              -> factory/motor1/alert
9. REST API                -> /api/status and /api/history
10. Live browser dashboard -> graphs, status, risk gauge, recent history

NO INTERNET IS REQUIRED BY THE DASHBOARD.
The graphs are drawn with built-in browser JavaScript; no Chart.js/CDN is used.

WINDOWS / CONDA SETUP
---------------------
Open Anaconda Prompt or Command Prompt where conda works:

    conda create -n edge_demo python=3.11 -y
    conda activate edge_demo
    pip install -r requirements.txt

Mosquitto is a separate application. If it is already running as a Windows service,
do NOT start another broker on port 1883.

VERIFY MQTT (optional)
----------------------
Subscriber terminal:

    "C:\Program Files\mosquitto\mosquitto_sub.exe" -h localhost -t "factory/motor1/#" -v

Publisher test from another terminal:

    "C:\Program Files\mosquitto\mosquitto_pub.exe" -h localhost -t "test/topic" -m "Hello class"

RUN THE DEMO
------------
Terminal A — edge gateway / web dashboard:

    conda activate edge_demo
    python app.py

Terminal B — simulated motor sensor:

    conda activate edge_demo
    python sensor.py --mode auto

Optional Terminal C — watch all MQTT motor messages:

    "C:\Program Files\mosquitto\mosquitto_sub.exe" -h localhost -t "factory/motor1/#" -v

Open in a browser:

    http://127.0.0.1:5000

REST endpoints:

    http://127.0.0.1:5000/api/status
    http://127.0.0.1:5000/api/history

CLASSROOM MODES
---------------
Automatic cycle:

    python sensor.py --mode auto

Normal motor:

    python sensor.py --mode normal

Warning condition:

    python sensor.py --mode warning

Fault condition:

    python sensor.py --mode fault

For a very clear live demonstration:
1. Start normal mode and show stable low vibration.
2. Stop sensor.py with Ctrl+C.
3. Run fault mode.
4. Watch raw vibration rise immediately.
5. Watch rolling RMS rise over several samples.
6. Watch NORMAL -> WARNING -> FAULT.
7. Show the MQTT alert in the subscriber terminal.
8. Open /api/status to show REST request/response.
9. Open /api/history to show data persisted in SQLite.

ANALYTICS MATH
--------------
The dashboard now uses a REAL rolling RMS of the latest 8 vibration readings:

    RMS = sqrt((x1^2 + x2^2 + ... + xN^2) / N)

Thresholds:

    RMS < 0.8            -> NORMAL  -> fault probability 5%
    0.8 <= RMS < 1.5     -> WARNING -> fault probability 45%
    RMS >= 1.5           -> FAULT   -> fault probability 92%

The probability values are intentionally simple classroom values, not an ML model.
You can explain that this rule block can later be replaced by SVM/CNN/LSTM/etc.

IMPORTANT TEACHING POINTS
-------------------------
MQTT = move/push telemetry
SQLite = persist/remember telemetry
REST = browser/client asks for current or historical data
Edge analytics = calculate RMS and make the decision locally
Dashboard = human-machine interface; it does NOT make the core decision

ARCHITECTURE
------------

sensor.py
   |
   | MQTT publish: factory/motor1/telemetry
   v
Mosquitto broker
   |
   v
app.py (EDGE GATEWAY)
   |-- MQTT subscriber
   |-- rolling RMS analytics
   |-- NORMAL/WARNING/FAULT decision
   |-- SQLite motor.db
   |-- MQTT fault alert
   |-- REST API
   v
Browser dashboard

In a real deployment, sensor.py can be replaced with an accelerometer/temperature
sensor and the laptop can be replaced with a Raspberry Pi, Jetson, or industrial gateway.
