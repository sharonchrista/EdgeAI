"""
extension_a_full_int8.py
Lab 1 - Extension A: Full Integer (INT8) Quantization.
Both weights AND activations converted to INT8, using a representative
dataset for calibration. Compares size, latency, and accuracy against
the dynamic-range quantized model from Step 2.
"""
import tensorflow as tf
import numpy as np
import time, os, statistics

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# 1. Load the trained Keras model and MNIST test data
model = tf.keras.models.load_model('mnist_model.keras')
print('Loaded model from  mnist_model.keras')

# Reuse the same MNIST loading approach as build_model.py.
# On your laptop this works directly; only needed the gz mirror in the sandbox.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_norm = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 2. Representative dataset generator for calibration
# The converter runs ~100-500 samples through the float model to observe
# the actual range of activation values at every layer, then picks fixed
# INT8 scales that cover that range. Too few samples -> poor calibration;
# too many -> slow conversion for no extra benefit. 200 is a solid default.
def representative_dataset():
    for i in range(200):
        sample = x_test_norm[i:i+1]
        yield [sample]

# 3. Convert with FULL INTEGER quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Force the converter to reject any op it cannot fully quantize to INT8,
# rather than silently falling back to float for that op.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8_model = converter.convert()
with open('model_int8_full.tflite', 'wb') as f:
    f.write(tflite_int8_model)

int8_size_kb = os.path.getsize('model_int8_full.tflite') / 1024
dynamic_size_kb = os.path.getsize('model.tflite') / 1024 if os.path.exists('model.tflite') else None

print(f'\n✔ Full INT8 model saved: model_int8_full.tflite  ({int8_size_kb:.1f} KB)')
if dynamic_size_kb:
    print(f'  Dynamic-range model (Step 2): {dynamic_size_kb:.1f} KB')
    print(f'  Size difference: {dynamic_size_kb - int8_size_kb:+.1f} KB')


# 4. Helper: benchmark latency for any .tflite model
def benchmark_latency(model_path, input_dtype, n_runs=100):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = x_test_norm[0:1]
    if input_dtype == np.int8:
        # Full INT8 models expect int8 input; rescale using the model's
        # own quantization parameters rather than a hardcoded formula.
        scale, zero_point = inp['quantization']
        x = (x / scale + zero_point).astype(np.int8)
    else:
        x = x.astype(np.float32)

    for _ in range(10):
        interp.set_tensor(inp['index'], x)
        interp.invoke()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), interp, inp, out


# 5. Helper: measure accuracy over N test images for any .tflite model
def measure_accuracy(model_path, input_dtype, n_images=1000):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    correct = 0
    for i in range(n_images):
        x = x_test_norm[i:i+1]
        if input_dtype == np.int8:
            scale, zero_point = inp['quantization']
            x = (x / scale + zero_point).astype(np.int8)
        else:
            x = x.astype(np.float32)
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        out_tensor = interp.get_tensor(out['index'])
        pred = int(np.argmax(out_tensor))
        if pred == y_test[i]:
            correct += 1
    return correct / n_images


# 6. Run comparison: dynamic-range (Step 2) vs full INT8 (this extension)
print('\n' + '=' * 65)
print('  QUANTIZATION COMPARISON')
print('=' * 65)

results = {}

if os.path.exists('model.tflite'):
    lat_dyn, *_ = benchmark_latency('model.tflite', np.float32)
    acc_dyn = measure_accuracy('model.tflite', np.float32, n_images=1000)
    results['Dynamic-range (Step 2)'] = {
        'size_kb': dynamic_size_kb, 'latency_ms': lat_dyn, 'accuracy': acc_dyn
    }

lat_int8, *_ = benchmark_latency('model_int8_full.tflite', np.int8)
acc_int8 = measure_accuracy('model_int8_full.tflite', np.int8, n_images=1000)
results['Full INT8 (this extension)'] = {
    'size_kb': int8_size_kb, 'latency_ms': lat_int8, 'accuracy': acc_int8
}

print(f'\n  {"Model":<28} {"Size (KB)":>10} {"Latency (ms)":>13} {"Accuracy":>10}')
print('  ' + '-' * 65)
for name, r in results.items():
    print(f'  {name:<28} {r["size_kb"]:>10.1f} {r["latency_ms"]:>13.4f} {r["accuracy"]*100:>9.2f}%')

if 'Dynamic-range (Step 2)' in results:
    d = results['Dynamic-range (Step 2)']
    i = results['Full INT8 (this extension)']
    print(f'\n  Size change    : {d["size_kb"] - i["size_kb"]:+.1f} KB '
          f'({(1 - i["size_kb"]/d["size_kb"])*100:+.1f}%)')
    print(f'  Latency change : {d["latency_ms"] - i["latency_ms"]:+.4f} ms '
          f'({(1 - i["latency_ms"]/d["latency_ms"])*100:+.1f}%)')
    print(f'  Accuracy delta : {(i["accuracy"] - d["accuracy"])*100:+.2f} percentage points')

print('\n✔ Measured over 1,000 test images')
print('  RECORD THESE VALUES: full INT8 should be similar or smaller in size,')
print('  comparable or faster latency, with a small (usually <1pt) accuracy drop.')