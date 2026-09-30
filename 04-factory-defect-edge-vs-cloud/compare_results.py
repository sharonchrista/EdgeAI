"""
compare_results.py
Lab 1 - Step 5: Side-by-side edge vs cloud comparison.
The cloud server must be running (python cloud_server.py).
"""
import numpy as np, time, os, requests, json
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \
    else np.random.rand(1, 28, 28, 1).astype(np.float32)
N = 100

# Edge benchmark
interp = Interpreter('model.tflite')
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
interp.set_tensor(inp['index'], x)
interp.invoke()  # warm-up
edge_t = []
for _ in range(N):
    t0 = time.perf_counter()
    interp.set_tensor(inp['index'], x)
    interp.invoke()
    edge_t.append((time.perf_counter() - t0) * 1000)
e_pred = int(np.argmax(interp.get_tensor(out['index'])))

# Cloud benchmark
URL = 'http://127.0.0.1:8000/predict'
payload = {'pixels': x.flatten().tolist()}
payload_bytes = len(json.dumps(payload).encode())
requests.post(URL, json=payload, timeout=10)  # warm-up
cloud_t = []
for _ in range(N):
    t0 = time.perf_counter()
    res = requests.post(URL, json=payload, timeout=10)
    cloud_t.append((time.perf_counter() - t0) * 1000)
c_pred = res.json()['prediction']

e, c = np.array(edge_t), np.array(cloud_t)
print('\n' + '=' * 65)
print(f'  {"METRIC":<28} {"EDGE":>16} {"CLOUD RTT":>16}')
print('=' * 65)
print(f'  {"Mean latency (ms)":<28} {e.mean():>16.3f} {c.mean():>16.3f}')
print(f'  {"Median (ms)":<28} {np.median(e):>16.3f} {np.median(c):>16.3f}')
print(f'  {"p95 latency (ms)":<28} {np.percentile(e, 95):>16.3f} {np.percentile(c, 95):>16.3f}')
print(f'  {"Throughput (inf/s)":<28} {1000/e.mean():>16.0f} {1000/c.mean():>16.0f}')
print(f'  {"Data sent per call":<28} {"0 bytes":>16} {payload_bytes:>13,} B')
print(f'  {"Network required":<28} {"No":>16} {"Yes":>16}')
print(f'  {"Prediction":<28} {e_pred:>16} {c_pred:>16}')
print('=' * 65)
print(f'\n  Edge speedup             : {c.mean()/e.mean():.1f}x faster than cloud RTT')
print(f'  Bandwidth per inference  : {payload_bytes:,} bytes  ({payload_bytes/1024:.1f} KB)')
print(f'  At 30 FPS bandwidth need : {payload_bytes*30/1024:.0f} KB/s  ({payload_bytes*30/1024/1024:.2f} MB/s)')
print('=' * 65)