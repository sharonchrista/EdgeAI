"""
step1_latency_sweep.py
Lab 2 - Step 1: Latency benchmark across thread counts.
Simulates MCU-tier (1 thread), Gateway-tier (2-4 threads), Edge-server (8 threads).
"""
import time, argparse, statistics, json, os
import numpy as np
import psutil
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH  = 'model.tflite'
WARMUP_RUNS = 10
BENCH_RUNS  = 200

TIER_LABELS = {
    1: 'MCU-tier (simulated: 1 thread)',
    2: 'Low-power gateway (2 threads)',
    4: 'Quad-core gateway / RPi4 (4 threads)',
    8: 'Edge server / Jetson (8 threads)',
}


def benchmark_one_config(num_threads):
    """Run the full benchmark for a given thread count. Returns stats dict."""
    interp = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = np.random.rand(*tuple(inp['shape'])).astype(inp['dtype'])

    # Warm-up runs (excluded from measurement)
    for _ in range(WARMUP_RUNS):
        interp.set_tensor(inp['index'], x)
        interp.invoke()

    # Measurement runs
    proc = psutil.Process()
    times = []
    proc.cpu_percent(interval=None)  # reset CPU counter
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # -> ms
    cpu_pct = proc.cpu_percent(interval=0.1)
    sorted_t = sorted(times)
    n = len(sorted_t)
    stats = {
        'threads': num_threads,
        'tier': TIER_LABELS.get(num_threads, f'{num_threads}-thread'),
        'mean_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'p95_ms': sorted_t[int(0.95 * n)],
        'p99_ms': sorted_t[int(0.99 * n)],
        'min_ms': min(times),
        'max_ms': max(times),
        'stdev_ms': statistics.stdev(times),
        'cpu_pct': cpu_pct,
        'runs': BENCH_RUNS,
    }
    return stats


def main():
    global MODEL_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Thread counts to test (space-separated)')
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()
    MODEL_PATH = args.model

    print(f'Model       : {MODEL_PATH}')
    print(f'Warm-up     : {WARMUP_RUNS} runs (excluded)')
    print(f'Bench runs  : {BENCH_RUNS} per configuration')
    print(f'Thread sweep: {args.threads}')
    print()

    all_results = []
    print(f'  {"Threads":>8}  {"Tier":<40}  {"Mean ms":>9}  {"Median ms":>9}  {"p95 ms":>8}  {"p99 ms":>8}  {"CPU%":>6}')
    print('  ' + '-' * 100)
    for t in args.threads:
        s = benchmark_one_config(t)
        all_results.append(s)
        print(f'  {s["threads"]:>8}  {s["tier"]:<40}  {s["mean_ms"]:>9.2f}  '
              f'{s["median_ms"]:>9.2f}  {s["p95_ms"]:>8.2f}  {s["p99_ms"]:>8.2f}  '
              f'{s["cpu_pct"]:>6.1f}')

    os.makedirs('results', exist_ok=True)
    with open('results/latency_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n✔ Results saved to results/latency_sweep.json')
    print(f'  RECORD THESE VALUES IN YOUR LATENCY TABLE')
    print(f'  Key question: which thread count first satisfies a 33ms/30fps SLA?')


if __name__ == '__main__':
    main()