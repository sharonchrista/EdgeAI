"""
step2_concurrent_throughput.py
Lab 2 - Step 2: Concurrent multi-worker throughput benchmark.
Tests how many simultaneous inference streams the hardware can sustain.
"""
import time, multiprocessing as mp, statistics, json, os
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = 'model.tflite'
WARMUP     = 5
BENCH_RUNS = 30


def worker_fn(model_path, num_threads, runs, warmup, result_queue):
    """
    Each worker loads its OWN TFLite interpreter.
    TFLite interpreters are NOT thread-safe -- never share across threads/processes.
    """
    interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.rand(*tuple(inp['shape'])).astype(inp['dtype'])

    for _ in range(warmup):
        interp.set_tensor(inp['index'], x)
        interp.invoke()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    result_queue.put(times)


def run_concurrent_benchmark(n_workers, threads_per_worker=1):
    """Launch n_workers processes and measure aggregate throughput."""
    q = mp.Queue()
    procs = []
    start = time.perf_counter()
    for _ in range(n_workers):
        p = mp.Process(target=worker_fn,
                       args=(MODEL_PATH, threads_per_worker, BENCH_RUNS, WARMUP, q))
        p.start()
        procs.append(p)
    all_times = []
    for _ in procs:
        all_times.extend(q.get())
    for p in procs:
        p.join()
    wall_time_s = time.perf_counter() - start
    total_infs = n_workers * BENCH_RUNS
    throughput = total_infs / wall_time_s
    mean_lat = statistics.mean(all_times)
    p95_lat = sorted(all_times)[int(0.95 * len(all_times))]
    return {
        'workers': n_workers,
        'threads_each': threads_per_worker,
        'total_infs': total_infs,
        'wall_time_s': round(wall_time_s, 3),
        'throughput_ips': round(throughput, 2),
        'mean_lat_ms': round(mean_lat, 2),
        'p95_lat_ms': round(p95_lat, 2),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers_list', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='List of worker counts to test')
    parser.add_argument('--threads', type=int, default=1,
                        help='TFLite threads per worker')
    args = parser.parse_args()

    print('Concurrent Throughput Benchmark')
    print(f'  Model      : {MODEL_PATH}')
    print(f'  Bench runs : {BENCH_RUNS} per worker  |  Warmup: {WARMUP}')
    print(f'  Threads/worker: {args.threads}')
    print()

    header = f'  {"Workers":>8}  {"Throughput (inf/s)":>20}  {"Mean lat (ms)":>15}  {"p95 lat (ms)":>13}'
    print(header)
    print('  ' + '-' * 65)
    results = []
    for n in args.workers_list:
        r = run_concurrent_benchmark(n, args.threads)
        results.append(r)
        print(f'  {r["workers"]:>8}  {r["throughput_ips"]:>20.2f}  '
              f'{r["mean_lat_ms"]:>15.2f}  {r["p95_lat_ms"]:>13.2f}')

    os.makedirs('results', exist_ok=True)
    with open('results/throughput.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✔ Saved to results/throughput.json')
    print('  IDENTIFY: at what worker count does throughput stop increasing?')
    print('  That is your hardware saturation point.')