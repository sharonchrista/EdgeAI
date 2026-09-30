"""
extension_c_thermal_throttling.py
Lab 2 - Extension C: Thermal throttling detection under sustained load.
Runs continuous concurrent inference for 120 seconds, logging latency
every second, to identify whether/when clock throttling degrades
performance under prolonged CPU load.
"""
import time, json, os, statistics
import multiprocessing as mp
import numpy as np
import psutil
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = 'model.tflite'
DURATION_S = 120       # total sustained-load duration
N_WORKERS = 4           # concurrent workers (your Step 2 saturation-adjacent tier)
LOG_INTERVAL_S = 1.0    # log latency once per second


def worker_fn(model_path, duration_s, result_queue, worker_id):
    """
    Each worker runs inference continuously for duration_s seconds,
    recording one representative latency sample approximately every
    second, then reports its full time series back to the parent.
    """
    interp = Interpreter(model_path=model_path, num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.rand(*tuple(inp['shape'])).astype(inp['dtype'])

    # Warm-up
    for _ in range(5):
        interp.set_tensor(inp['index'], x)
        interp.invoke()

    samples = []  # list of (elapsed_seconds, latency_ms)
    start = time.perf_counter()
    last_log = start

    while (time.perf_counter() - start) < duration_s:
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        # Log once per second rather than every single inference
        # (which would run thousands of times over 120s)
        if t1 - last_log >= LOG_INTERVAL_S:
            elapsed = t1 - start
            samples.append((elapsed, latency_ms))
            last_log = t1

    result_queue.put((worker_id, samples))


def run_sustained_load():
    q = mp.Queue()
    procs = []
    for wid in range(N_WORKERS):
        p = mp.Process(target=worker_fn, args=(MODEL_PATH, DURATION_S, q, wid))
        p.start()
        procs.append(p)

    all_results = {}
    for _ in procs:
        wid, samples = q.get()
        all_results[wid] = samples
    for p in procs:
        p.join()

    return all_results


if __name__ == '__main__':
    print('=' * 65)
    print(f'  SUSTAINED LOAD THERMAL THROTTLING TEST')
    print(f'  Duration: {DURATION_S}s  |  Workers: {N_WORKERS}  |  Logging every {LOG_INTERVAL_S}s')
    print('=' * 65)
    print('\nThis will take exactly 2 minutes. Monitor Task Manager -> Performance')
    print('-> CPU to watch clock speed / temperature behavior live if you want.\n')

    t_start = time.perf_counter()
    results = run_sustained_load()
    wall_time = time.perf_counter() - t_start
    print(f'\nCompleted in {wall_time:.1f}s')

    # Aggregate: average latency across all workers at each ~1-second tick
    # Align by nearest-second bucket since workers won't log at identical instants
    buckets = {}  # {second_bucket: [latencies]}
    for wid, samples in results.items():
        for elapsed, lat in samples:
            bucket = int(round(elapsed))
            buckets.setdefault(bucket, []).append(lat)

    timeline = sorted(buckets.keys())
    mean_latencies = [statistics.mean(buckets[t]) for t in timeline]

    os.makedirs('results', exist_ok=True)
    with open('results/thermal_throttle.json', 'w') as f:
        json.dump({'timeline_s': timeline, 'mean_latency_ms': mean_latencies}, f, indent=2)

    # Simple throttle-point detection: compare early-window vs late-window mean
    early_window = mean_latencies[:15] if len(mean_latencies) >= 15 else mean_latencies[:len(mean_latencies)//2]
    late_window = mean_latencies[-15:] if len(mean_latencies) >= 15 else mean_latencies[len(mean_latencies)//2:]
    early_mean = statistics.mean(early_window)
    late_mean = statistics.mean(late_window)
    degradation_pct = (late_mean - early_mean) / early_mean * 100

    print(f'\n  Early-window mean latency (first ~15s): {early_mean:.3f} ms')
    print(f'  Late-window mean latency (last ~15s)  : {late_mean:.3f} ms')
    print(f'  Degradation                            : {degradation_pct:+.1f}%')

    if degradation_pct > 15:
        # Find approximate throttle onset: first sustained rise above
        # 1.15x the early-window baseline
        threshold = early_mean * 1.15
        throttle_point = None
        for t, lat in zip(timeline, mean_latencies):
            if lat > threshold:
                throttle_point = t
                break
        if throttle_point:
            print(f'  Likely throttling onset around t = {throttle_point}s')
        print('  -> Meaningful degradation detected: consistent with thermal throttling.')
    else:
        print('  -> No strong evidence of thermal throttling in this run')
        print('     (laptop cooling may be adequate for this workload, or the')
        print('     test duration/load was insufficient to trigger it).')

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timeline, mean_latencies, marker='o', markersize=3, linewidth=1.2, color='#1C7293')
    ax.axhline(early_mean, color='green', linestyle=':', linewidth=1.5,
               label=f'Early baseline ({early_mean:.2f} ms)')
    ax.set_xlabel('Elapsed Time (seconds)')
    ax.set_ylabel('Mean Latency Across Workers (ms)')
    ax.set_title(f'Latency Over 120s Sustained Load ({N_WORKERS} concurrent workers)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/thermal_throttle_timeline.png', dpi=120, bbox_inches='tight')
    plt.show()
    print(f'\n✔ Timeline chart saved to results/thermal_throttle_timeline.png')
    print(f'✔ Raw data saved to results/thermal_throttle.json')