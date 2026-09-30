"""
step2_sensor_publisher.py
Lab 3 - Step 2: MQTT sensor publisher.
Simulates a refrigerated truck's two sensors -- temperature probe and
compressor vibration accelerometer -- publishing to separate MQTT topics.
Includes a scripted scenario that drifts into a Critical state partway
through, to test the subscriber's alerting logic later.
"""
import paho.mqtt.client as mqtt
import json
import time
import random
import argparse

BROKER = 'localhost'
PORT = 1883
TOPIC_TEMP = 'safevax/truck01/temperature'
TOPIC_VIB = 'safevax/truck01/vibration'
PUBLISH_INTERVAL_S = 1.0


def make_client():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='sensor_publisher')
    client.connect(BROKER, PORT, keepalive=60)
    return client


def generate_reading(t_elapsed, scenario):
    """
    Returns (temp_c, vib_au) for the current elapsed time, following one
    of three scenarios:
      normal    -- stays in the healthy range the whole run
      drifting  -- starts normal, gradually drifts to Critical by the end
      critical  -- starts already in the Critical range
    """
    if scenario == 'normal':
        temp = random.uniform(2.0, 8.0)
        vib = abs(random.gauss(0.5, 0.3))

    elif scenario == 'drifting':
        # Linear drift from Normal range to Critical range over 60 seconds
        progress = min(t_elapsed / 60.0, 1.0)  # 0 -> 1 over 60s
        temp = 4.0 + progress * 12.0   # 4C -> 16C
        vib = 0.5 + progress * 4.0     # 0.5 -> 4.5 AU
        # Add small noise so it's not a perfectly straight line
        temp += random.uniform(-0.3, 0.3)
        vib += random.uniform(-0.15, 0.15)

    elif scenario == 'critical':
        temp = random.uniform(13.0, 18.0)
        vib = random.uniform(3.5, 5.5)

    else:
        raise ValueError(f'Unknown scenario: {scenario}')

    return round(temp, 2), round(vib, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['normal', 'drifting', 'critical'],
                        default='drifting',
                        help='normal=stays healthy, drifting=degrades over 60s, '
                             'critical=starts already critical')
    parser.add_argument('--duration', type=int, default=90,
                        help='total seconds to publish')
    args = parser.parse_args()

    client = make_client()
    client.loop_start()

    print(f'Publisher started  |  scenario={args.scenario}  |  '
          f'duration={args.duration}s  |  interval={PUBLISH_INTERVAL_S}s')
    print(f'Publishing to:')
    print(f'  {TOPIC_TEMP}')
    print(f'  {TOPIC_VIB}')
    print('Press Ctrl+C to stop early.\n')

    start = time.time()
    reading_count = 0
    try:
        while (time.time() - start) < args.duration:
            elapsed = time.time() - start
            temp, vib = generate_reading(elapsed, args.scenario)

            temp_payload = json.dumps({
                'sensor': 'temperature',
                'truck_id': 'truck01',
                'value_c': temp,
                'timestamp': time.time()
            })
            vib_payload = json.dumps({
                'sensor': 'vibration',
                'truck_id': 'truck01',
                'value_au': vib,
                'timestamp': time.time()
            })

            client.publish(TOPIC_TEMP, temp_payload, qos=1)
            client.publish(TOPIC_VIB, vib_payload, qos=1)

            reading_count += 1
            print(f'  [{elapsed:5.1f}s] temp={temp:5.2f}C  vib={vib:5.3f}AU  '
                  f'(reading #{reading_count})')

            time.sleep(PUBLISH_INTERVAL_S)

    except KeyboardInterrupt:
        print('\nStopped by user.')

    print(f'\nTotal readings published: {reading_count}')
    client.loop_stop()
    client.disconnect()


if __name__ == '__main__':
    main()