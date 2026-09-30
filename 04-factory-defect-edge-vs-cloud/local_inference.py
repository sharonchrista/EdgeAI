"""
local_inference.py
Lab 1 - Step 3: Edge inference benchmark using TFLite.
"""
import numpy as np
import time, os

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# 1. Load model
interpreter = Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f'Input shape  : {inp["shape"]}')
print(f'Output shape : {out["shape"]}')

# 2. Load test sample
x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \
    else np.random.rand(1, 28, 28, 1).astype(np.float32)

# 3. Warm-up (JIT compilation, not measured)
interpreter.set_tensor(inp['index'], x)
interpreter.invoke()

# 4. Benchmark: 100 inferences
N = 100
times = []
for _ in range(N):
    t0 = time.perf_counter()
    interpreter.set_tensor(inp['index'], x)
    interpreter.invoke()
    times.append((time.perf_counter() - t0) * 1000)
times = np.array(times)
pred = int(np.argmax(interpreter.get_tensor(out['index'])))
conf = float(np.max(interpreter.get_tensor(out['index'])))

# 5. Report
print('\n' + '=' * 50)
print('  EDGE INFERENCE RESULTS')
print('=' * 50)
print(f'  Predicted class  : {pred}')
print(f'  Confidence       : {conf:.4f}')
print(f'  Mean latency     : {times.mean():.3f} ms')
print(f'  Median latency   : {np.median(times):.3f} ms')
print(f'  p95 latency      : {np.percentile(times, 95):.3f} ms')
print(f'  Max latency      : {times.max():.3f} ms')
print(f'  Throughput       : {1000/times.mean():.0f} inferences/sec')
print('=' * 50)
print('  RECORD THESE VALUES IN YOUR RESULTS TABLE')