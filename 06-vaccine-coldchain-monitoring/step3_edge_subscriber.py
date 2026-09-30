"""
step3_edge_subscriber.py
Lab 3 - Step 3: Edge subscriber node.
Subscribes to both sensor topics, fuses the latest readings, normalises
using the EXACT stats saved in Step 1, runs TFLite inference, logs every
decision to SQLite, and raises an alert after N consecutive Critical
predictions.
"""
import paho.mqtt.client as mqtt
import json
import time
import sqlite3
import numpy as np
import os
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

BROKER = 'localhost'
PORT = 1883
TOPIC_TEMP = 'safevax/truck01/temperature'
TOPIC_VIB = 'safevax/truck01/vibration'

MODEL_PATH = 'anomaly_model.tflite'
NORM_STATS_PATH = 'normalisation_stats.npy'
DB_PATH = 'edge_data.db'

CONSECUTIVE_CRITICAL_THRESHOLD = 3  # N consecutive Critical readings -> alert
CLASS_LABELS = ['Normal', 'Warning', 'Critical']

# --- Load model and normalisation stats once at startup ---
interp = Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

norm_stats = np.load(NORM_STATS_PATH)
MEAN, STD = norm_stats[0], norm_stats[1]
print(f'Loaded model and normalisation stats:')
print(f'  MEAN = {MEAN}')
print(f'  STD  = {STD}')

# --- SQLite setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            temperature_c REAL,
            vibration_au REAL,
            predicted_class TEXT,
            confidence REAL,
            is_alert INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def log_inference(timestamp, temp, vib, pred_class, confidence, is_alert):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO inferences (timestamp, temperature_c, vibration_au,
                                 predicted_class, confidence, is_alert)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, temp, vib, pred_class, confidence, int(is_alert)))
    conn.commit()
    conn.close()


# --- Sensor fusion state: latest reading from each topic ---
latest_readings = {'temperature': None, 'vibration': None}
consecutive_critical_count = 0


def run_inference(temp_c, vib_au):
    """Normalise using saved stats, run TFLite inference, return (class, confidence)."""
    raw = np.array([[temp_c, vib_au]], dtype=np.float32)
    normed = ((raw - MEAN) / STD).astype(np.float32)

    interp.set_tensor(inp_detail['index'], normed)
    interp.invoke()
    output = interp.get_tensor(out_detail['index'])[0]
    pred_idx = int(np.argmax(output))
    confidence = float(output[pred_idx])
    return CLASS_LABELS[pred_idx], confidence


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f'Connected to broker (reason_code={reason_code})')
    client.subscribe(TOPIC_TEMP, qos=1)
    client.subscribe(TOPIC_VIB, qos=1)
    print(f'Subscribed to {TOPIC_TEMP} and {TOPIC_VIB}\n')


def on_message(client, userdata, msg):
    global consecutive_critical_count

    payload = json.loads(msg.payload.decode())

    if msg.topic == TOPIC_TEMP:
        latest_readings['temperature'] = payload['value_c']
    elif msg.topic == TOPIC_VIB:
        latest_readings['vibration'] = payload['value_au']

    # Only run inference once we have BOTH sensor readings available.
    # This is the sensor fusion step -- neither sensor alone is sufficient.
    if latest_readings['temperature'] is None or latest_readings['vibration'] is None:
        return

    temp = latest_readings['temperature']
    vib = latest_readings['vibration']

    pred_class, confidence = run_inference(temp, vib)
    timestamp = time.time()

    is_alert = False
    if pred_class == 'Critical':
        consecutive_critical_count += 1
    else:
        consecutive_critical_count = 0

    if consecutive_critical_count >= CONSECUTIVE_CRITICAL_THRESHOLD:
        is_alert = True

    log_inference(timestamp, temp, vib, pred_class, confidence, is_alert)

    status_marker = '🚨 ALERT' if is_alert else ''
    print(f'  temp={temp:5.2f}C  vib={vib:5.3f}AU  ->  {pred_class:<8} '
          f'(conf={confidence:.3f})  '
          f'[consecutive_critical={consecutive_critical_count}]  {status_marker}')

    if is_alert:
        print(f'  {"="*60}')
        print(f'  ALERT: {CONSECUTIVE_CRITICAL_THRESHOLD} consecutive Critical readings.')
        print(f'  In production: notify driver dashboard, log to fleet system.')
        print(f'  {"="*60}\n')


def main():
    init_db()
    print(f'SQLite database ready: {DB_PATH}\n')

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='edge_subscriber')
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, keepalive=60)
    print('Edge subscriber listening. Press Ctrl+C to stop.\n')

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print('\nStopped by user.')
        client.disconnect()


if __name__ == '__main__':
    main()