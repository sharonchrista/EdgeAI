"""
extension_c_fp32_vs_int8.py
Lab 1 - Extension C: Full comparison of FP32 (no optimization) vs
dynamic-range INT8 quantization, evaluated on the FULL 10,000-image
MNIST test set. Produces a 3-panel bar chart: accuracy, latency, size.
"""
import tensorflow as tf
import numpy as np
import time, os, statistics
import matplotlib.pyplot as plt

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# 1. Load trained model and the FULL MNIST test set (10,000 images)
model = tf.keras.models.load_model('mnist_model.keras')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_norm = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f'Full test set loaded: {len(x_test_norm):,} images')

# 2. Convert (1) FP32 baseline -- no optimizations at all
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
# No converter.optimizations line -- this is the deliberate FP32 baseline
tflite_fp32 = converter_fp32.convert()
with open('model_fp32.tflite', 'wb') as f:
    f.write(tflite_fp32)

# 3. Convert (2) dynamic-range INT8 (same as Step 2)
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_int8 = converter_int8.convert()
with open('model_dynamic_int8.tflite', 'wb') as f:
    f.write(tflite_int8)

fp32_size_kb = os.path.getsize('model_fp32.tflite') / 1024
int8_size_kb = os.path.getsize('model_dynamic_int8.tflite') / 1024
print(f'\nFP32 model size       : {fp32_size_kb:.1f} KB')
print(f'Dynamic INT8 model size: {int8_size_kb:.1f} KB')


# 4. Helper: full-test-set accuracy + mean latency for a given .tflite model
def evaluate_full(model_path, n_latency_runs=100):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # --- Accuracy over all 10,000 test images ---
    correct = 0
    for i in range(len(x_test_norm)):
        x = x_test_norm[i:i+1].astype(inp['dtype'])
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out['index'])))
        if pred == y_test[i]:
            correct += 1
    accuracy = correct / len(x_test_norm)

    # --- Mean latency over n_latency_runs on one fixed sample ---
    x_bench = x_test_norm[0:1].astype(inp['dtype'])
    for _ in range(10):  # warm-up
        interp.set_tensor(inp['index'], x_bench)
        interp.invoke()
    times = []
    for _ in range(n_latency_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x_bench)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    mean_latency_ms = statistics.mean(times)

    return accuracy, mean_latency_ms


# 5. Evaluate both models on the full test set
print('\nEvaluating FP32 on all 10,000 test images (this takes a bit longer)...')
acc_fp32, lat_fp32 = evaluate_full('model_fp32.tflite')

print('Evaluating Dynamic INT8 on all 10,000 test images...')
acc_int8, lat_int8 = evaluate_full('model_dynamic_int8.tflite')

# 6. Report
print('\n' + '=' * 65)
print('  FP32 vs DYNAMIC-RANGE INT8 -- FULL 10,000-IMAGE TEST SET')
print('=' * 65)
print(f'  {"Model":<20} {"Size (KB)":>10} {"Latency (ms)":>13} {"Top-1 Acc":>10}')
print('  ' + '-' * 60)
print(f'  {"FP32 (baseline)":<20} {fp32_size_kb:>10.1f} {lat_fp32:>13.4f} {acc_fp32*100:>9.2f}%')
print(f'  {"Dynamic INT8":<20} {int8_size_kb:>10.1f} {lat_int8:>13.4f} {acc_int8*100:>9.2f}%')

acc_delta_pts = (acc_int8 - acc_fp32) * 100
size_reduction_pct = (1 - int8_size_kb / fp32_size_kb) * 100
latency_change_pct = (1 - lat_int8 / lat_fp32) * 100 if lat_fp32 > 0 else 0

print(f'\n  Accuracy delta   : {acc_delta_pts:+.3f} percentage points')
print(f'  Size reduction   : {size_reduction_pct:+.1f}%')
print(f'  Latency change   : {latency_change_pct:+.1f}%')

# 7. Bar chart -- 3 panels: accuracy, latency, size
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('FP32 vs Dynamic-Range INT8 -- Full 10,000-Image MNIST Test Set',
             fontsize=12, fontweight='bold')
labels = ['FP32\n(baseline)', 'Dynamic\nINT8']
colors = ['#1C7293', '#02C39A']

# Panel 1: Accuracy
accs = [acc_fp32 * 100, acc_int8 * 100]
bars0 = axes[0].bar(labels, accs, color=colors, edgecolor='white', alpha=0.9)
axes[0].set_ylabel('Top-1 Accuracy (%)')
axes[0].set_title('Accuracy (10,000 images)')
axes[0].set_ylim(min(accs) - 1, 100)
for bar, val in zip(bars0, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Panel 2: Latency
lats = [lat_fp32, lat_int8]
bars1 = axes[1].bar(labels, lats, color=colors, edgecolor='white', alpha=0.9)
axes[1].set_ylabel('Mean Latency (ms)')
axes[1].set_title('Inference Latency')
for bar, val in zip(bars1, lats):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lats)*0.02,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Panel 3: Size
sizes = [fp32_size_kb, int8_size_kb]
bars2 = axes[2].bar(labels, sizes, color=colors, edgecolor='white', alpha=0.9)
axes[2].set_ylabel('Model Size (KB)')
axes[2].set_title('File Size')
for bar, val in zip(bars2, sizes):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/fp32_vs_int8_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print('\n✔ Chart saved to results/fp32_vs_int8_comparison.png')