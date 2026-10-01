#!/usr/bin/env python3
"""Read a DHT11 or DHT22 on Raspberry Pi GPIO4 (physical pin 7).

Use a 3.3 V supply, common ground, and a 4.7–10 kΩ DATA pull-up for
bare sensors. Install adafruit-circuitpython-dht in a Python virtualenv.
"""
import argparse
import time

import board
import adafruit_dht


def main():
    parser = argparse.ArgumentParser(description="Read a DHT sensor on GPIO4")
    parser.add_argument("--sensor", choices=("dht11", "dht22"), default="dht22")
    args = parser.parse_args()
    sensor_type = adafruit_dht.DHT11 if args.sensor == "dht11" else adafruit_dht.DHT22
    sensor = sensor_type(board.D4, use_pulseio=False)
    print(f"Reading {args.sensor.upper()} on GPIO4 every 2 seconds. Ctrl+C to stop.")
    try:
        while True:
            try:
                temperature = sensor.temperature
                humidity = sensor.humidity
                if temperature is None or humidity is None:
                    print("No complete reading; retrying.")
                else:
                    print(f"{temperature:.1f} C | {humidity:.1f} % RH")
            except RuntimeError as error:
                print(f"Transient read error: {error}; retrying.")
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sensor.exit()


if __name__ == "__main__":
    main()
