"""
cloud_client.py
Lab 1 - Step 4b: Cloud benchmark client. Measures HTTP round-trip time (RTT).
"""
import numpy as np
import requests
import time
import json
import os

x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \
    else np.random.rand(1, 28, 28, 1).astype(np.float32)
pixels = x.flatten().tolist()

url = "http://127.0.0.1:8000/predict"
payload = {"pixels": pixels}

# Warm-up
requests.post(url, json=payload)

N = 100
times = []
for _ in range(N):
    t0 = time.perf_counter()
    r = requests.post(url, json=payload)
    times.append((time.perf_counter() - t0) * 1000)
times = np.array(times)
result = r.json()

# Bandwidth estimate
payload_bytes = len(json.dumps(payload).encode('utf-8'))

print('\n' + '=' * 50)
print('  CLOUD (REST API) INFERENCE RESULTS')
print('=' * 50)
print(f'  Predicted class  : {result["prediction"]}')
print(f'  Confidence       : {result["confidence"]:.4f}')
print(f'  Mean RTT         : {times.mean():.3f} ms')
print(f'  Median RTT       : {np.median(times):.3f} ms')
print(f'  p95 RTT          : {np.percentile(times, 95):.3f} ms')
print(f'  Max RTT          : {times.max():.3f} ms')
print(f'  Throughput       : {1000/times.mean():.0f} req/sec')
print(f'  Payload size     : {payload_bytes:,} bytes (JSON, one image)')
print('=' * 50)