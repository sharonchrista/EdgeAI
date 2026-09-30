"""
extension_b_concurrent_cloud.py
Lab 1 - Extension B: Concurrent Cloud Requests.
Simulates 4 simultaneous cameras hitting one cloud inference endpoint,
using asyncio + httpx, and compares against the single-request baseline.
"""
import asyncio
import httpx
import numpy as np
import time
import json
import os
import statistics

URL = "http://127.0.0.1:8000/predict"
N_CAMERAS = 4          # concurrent simulated cameras
ROUNDS = 25             # rounds of N_CAMERAS simultaneous requests each
                        # -> total requests = N_CAMERAS * ROUNDS = 100, matching Step 4's N=100

x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \
    else np.random.rand(1, 28, 28, 1).astype(np.float32)
payload = {"pixels": x.flatten().tolist()}
payload_bytes = len(json.dumps(payload).encode('utf-8'))


async def single_request(client):
    t0 = time.perf_counter()
    r = await client.post(URL, json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, r.json()


async def one_round(client, n_concurrent):
    """Fire n_concurrent requests at the same instant, wait for all to finish."""
    tasks = [single_request(client) for _ in range(n_concurrent)]
    results = await asyncio.gather(*tasks)
    return results  # list of (latency_ms, response_json)


async def run_concurrent_benchmark():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Warm-up
        await single_request(client)

        all_latencies = []
        wall_start = time.perf_counter()
        for _ in range(ROUNDS):
            round_results = await one_round(client, N_CAMERAS)
            all_latencies.extend([lat for lat, _ in round_results])
        wall_time_s = time.perf_counter() - wall_start

    total_requests = N_CAMERAS * ROUNDS
    throughput = total_requests / wall_time_s
    return all_latencies, wall_time_s, total_requests, throughput


def run_sequential_baseline(n_requests=100):
    """Single-request-at-a-time baseline, using requests (sync), matching Step 4."""
    import requests
    requests.post(URL, json=payload)  # warm-up
    times = []
    t_start = time.perf_counter()
    for _ in range(n_requests):
        t0 = time.perf_counter()
        r = requests.post(URL, json=payload)
        times.append((time.perf_counter() - t0) * 1000)
    wall_time_s = time.perf_counter() - t_start
    throughput = n_requests / wall_time_s
    return times, wall_time_s, throughput


if __name__ == '__main__':
    print(f'Simulating {N_CAMERAS} concurrent cameras x {ROUNDS} rounds '
          f'= {N_CAMERAS * ROUNDS} total requests')
    print(f'Payload size per request: {payload_bytes:,} bytes\n')

    # 1. Sequential baseline (Step 4 equivalent, re-measured here for a fair
    #    same-session comparison)
    print('Running sequential baseline (1 request at a time)...')
    seq_times, seq_wall_s, seq_throughput = run_sequential_baseline(n_requests=100)
    seq_arr = np.array(seq_times)

    # 2. Concurrent benchmark
    print(f'Running concurrent benchmark ({N_CAMERAS} simultaneous requests)...')
    conc_latencies, conc_wall_s, conc_total, conc_throughput = asyncio.run(
        run_concurrent_benchmark()
    )
    conc_arr = np.array(conc_latencies)

    # 3. Report
    print('\n' + '=' * 70)
    print(f'  {"Metric":<32} {"Sequential (1x)":>16} {"Concurrent (4x)":>16}')
    print('=' * 70)
    print(f'  {"Total requests":<32} {100:>16} {conc_total:>16}')
    print(f'  {"Wall time (s)":<32} {seq_wall_s:>16.3f} {conc_wall_s:>16.3f}')
    print(f'  {"Mean per-request latency (ms)":<32} {seq_arr.mean():>16.3f} {conc_arr.mean():>16.3f}')
    print(f'  {"p95 per-request latency (ms)":<32} {np.percentile(seq_arr,95):>16.3f} {np.percentile(conc_arr,95):>16.3f}')
    print(f'  {"Max latency (ms)":<32} {seq_arr.max():>16.3f} {conc_arr.max():>16.3f}')
    print(f'  {"Throughput (req/sec)":<32} {seq_throughput:>16.2f} {conc_throughput:>16.2f}')
    print('=' * 70)

    print(f'\n  Per-request latency inflation under concurrency: '
          f'{(conc_arr.mean() / seq_arr.mean() - 1) * 100:+.1f}%')
    print(f'  Throughput change under concurrency: '
          f'{(conc_throughput / seq_throughput - 1) * 100:+.1f}%')
    print(f'\n  Compare against edge inference (Step 3/5): edge latency does not')
    print(f'  degrade under concurrent load in the same way, since each device')
    print(f'  runs its own independent inference with no shared server bottleneck.')