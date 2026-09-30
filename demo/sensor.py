import argparse
import json
import random
import time
from datetime import datetime
import paho.mqtt.client as mqtt

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC = "factory/motor1/telemetry"

parser = argparse.ArgumentParser(description="Simulated motor sensor")
parser.add_argument("--mode", choices=["normal", "warning", "fault", "auto"], default="auto")
parser.add_argument("--interval", type=float, default=1.0)
args = parser.parse_args()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="motor-sensor")
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.loop_start()

counter = 0
try:
    while True:
        counter += 1
        mode = args.mode
        if mode == "auto":
            # Classroom-friendly cycle: 12 normal, 8 warning, 10 fault, then repeat.
            phase = counter % 30
            mode = "normal" if phase < 12 else "warning" if phase < 20 else "fault"

        if mode == "normal":
            temperature = random.uniform(62, 74)
            vibration = random.uniform(0.25, 0.70)
        elif mode == "warning":
            temperature = random.uniform(74, 84)
            vibration = random.uniform(0.85, 1.35)
        else:
            temperature = random.uniform(84, 96)
            vibration = random.uniform(1.7, 2.5)

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "machine": "motor1",
            "temperature": temperature,
            "vibration": vibration,
        }
        client.publish(TOPIC, json.dumps(payload), qos=1)
        print(f"[SENSOR] {mode.upper():7s}  T={temperature:.2f} C  V={vibration:.2f} g")
        time.sleep(args.interval)
except KeyboardInterrupt:
    pass
finally:
    client.loop_stop()
    client.disconnect()
