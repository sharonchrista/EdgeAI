"""
extension_d_batched_inference.py
Lab 2 - Extension D: Batched inference throughput benchmark.
Tests batch sizes 1, 2, 4, 8 -- measures per-image latency, throughput,
and throughput-per-watt, to determine optimal batch size for SkyRoute.
"""
import time, json, os, statistics
import numpy as np
import tensorflow as tf
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = 'model.tflite'
BATCH_SIZES = [1, 2, 4, 8]
WARMUP_RUNS = 5
BENCH_RUNS = 30

# Reuse Step 3's power profile for the throughput-per-watt calculation.
# SkyRoute drone realistically runs on gateway-tier hardware (per your
# Lab 2 R5 hardware recommendation), so we use that power figure here.
ASSUMED_POWER_W = 4.0  # gateway-tier, matches step3_energy_analysis.py


def build_batched_model(batch_size):
    """
    TFLite models are usually exported with a fixed batch dimension of 1.
    To test batch_size > 1, we rebuild the model from the original Keras
    model with an explicit batch input shape, then reconvert to TFLite.
    """
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', input_shape=(224, 224, 3))

    # Wrap with an explicit batch-size input to force the converter to
    # bake in that batch dimension rather than leaving it dynamic.
    inp = tf.keras.Input(shape=(224, 224, 3), batch_size=batch_size)
    out = base_model(inp)
    batched_model = tf.keras.Model(inp, out)

    converter = tf.lite.TFLiteConverter.from_keras_model(batched_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    path = f'model_batch{batch_size}.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    return path


def benchmark_batch(model_path, batch_size, n_runs=BENCH_RUNS):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    x = np.random.rand(*tuple(inp_detail['shape'])).astype(inp_detail['dtype'])

    for _ in range(WARMUP_RUNS):
        interp.set_tensor(inp_detail['index'], x)
        interp.invoke()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp_detail['index'], x)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)  # ms per invoke() call

    mean_batch_latency_ms = statistics.mean(times)
    mean_per_image_latency_ms = mean_batch_latency_ms / batch_size
    images_per_sec = (batch_size * 1000) / mean_batch_latency_ms

    return {
        'batch_size': batch_size,
        'mean_batch_latency_ms': round(mean_batch_latency_ms, 4),
        'mean_per_image_latency_ms': round(mean_per_image_latency_ms, 4),
        'throughput_ips': round(images_per_sec, 2),
    }


if __name__ == '__main__':
    print('=' * 70)
    print('  BATCHED INFERENCE THROUGHPUT BENCHMARK')
    print(f'  Batch sizes: {BATCH_SIZES}  |  {BENCH_RUNS} runs each after {WARMUP_RUNS} warmup')
    print('=' * 70)

    results = []
    for bs in BATCH_SIZES:
        print(f'\nBuilding and converting batch_size={bs} model...')
        model_path = build_batched_model(bs)
        size_kb = os.path.getsize(model_path) / 1024
        print(f'  Saved {model_path} ({size_kb:.1f} KB)')

        print(f'  Benchmarking...')
        r = benchmark_batch(model_path, bs)
        results.append(r)
        print(f'  batch={bs}: {r["throughput_ips"]:.1f} img/s, '
              f'{r["mean_per_image_latency_ms"]:.3f} ms/image')

    # Throughput-per-watt (assumes constant power draw regardless of batch
    # size -- a simplification, since larger batches may draw marginally
    # more power, but this isolates the throughput effect cleanly)
    for r in results:
        r['throughput_per_watt'] = round(r['throughput_ips'] / ASSUMED_POWER_W, 2)

    print('\n' + '=' * 70)
    print(f'  {"Batch":>6} {"Batch lat(ms)":>14} {"Per-img lat(ms)":>16} '
          f'{"Throughput(img/s)":>18} {"img/s/W":>9}')
    print('  ' + '-' * 68)
    for r in results:
        print(f'  {r["batch_size"]:>6} {r["mean_batch_latency_ms"]:>14.3f} '
              f'{r["mean_per_image_latency_ms"]:>16.3f} '
              f'{r["throughput_ips"]:>18.1f} {r["throughput_per_watt"]:>9.1f}')

    # Identify optimal batch size (highest throughput-per-watt)
    best = max(results, key=lambda r: r['throughput_per_watt'])
    baseline = results[0]  # batch=1
    speedup_vs_b1 = best['throughput_ips'] / baseline['throughput_ips']

    print(f'\n  Best throughput/watt: batch={best["batch_size"]} '
          f'({best["throughput_per_watt"]} img/s/W)')
    print(f'  Speedup vs batch=1  : {speedup_vs_b1:.2f}x')

    os.makedirs('results', exist_ok=True)
    with open('results/batched_inference.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✔ Results saved to results/batched_inference.json')

    print(f'\n  Is MobileNetV2 batch-efficient on this hardware?')
    print(f'  Compare throughput_ips growth vs batch_size growth above --')
    print(f'  near-linear growth = batch-efficient; sublinear/flat = not.')