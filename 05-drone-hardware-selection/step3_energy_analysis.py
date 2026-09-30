"""
step3_energy_analysis.py
Lab 2 - Step 3: Energy per inference estimation and battery life calculation.
Uses E = Power(W) x Latency(s) for each hardware tier.
"""
import time, json, os, statistics
import numpy as np
import psutil
import matplotlib.pyplot as plt
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# Hardware power profiles (conservative peak estimates)
# Source: manufacturer datasheets for each hardware class
HARDWARE_PROFILES = {
    'MCU-tier (1 thread)': {'threads': 1, 'power_w': 0.5},     # STM32H7 class
    'Gateway-tier (4 thr)': {'threads': 4, 'power_w': 4.0},    # RPi 3 at full load
    'Edge-server (8 thr)': {'threads': 8, 'power_w': 25.0},    # Jetson Orin NX
}

MODEL_PATH = 'model.tflite'
WARMUP = 10
BENCH_RUNS = 100


def benchmark_latency(num_threads):
    interp = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.rand(*tuple(inp['shape'])).astype(inp['dtype'])
    for _ in range(WARMUP):
        interp.set_tensor(inp['index'], x)
        interp.invoke()
    times = []
    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    cpu_pct = proc.cpu_percent(interval=0.2)
    return statistics.mean(times), cpu_pct


print('Energy Per Inference Analysis')
print(f'  Model: {MODEL_PATH}  |  {BENCH_RUNS} runs after {WARMUP} warmup')
print()

energy_results = []
print(f'  {"Hardware Tier":<26}  {"Threads":>8}  {"Power (W)":>10}  {"Lat (ms)":>10}  {"Energy (mJ)":>12}  {"inf/Wh":>10}')
print('  ' + '-' * 85)
for tier_name, profile in HARDWARE_PROFILES.items():
    mean_ms, cpu_pct = benchmark_latency(profile['threads'])
    energy_j = profile['power_w'] * (mean_ms / 1000)
    energy_mj = energy_j * 1000
    infs_per_wh = 3600 / energy_j if energy_j > 0 else 0
    row = {
        'tier': tier_name,
        'threads': profile['threads'],
        'power_w': profile['power_w'],
        'mean_ms': round(mean_ms, 2),
        'energy_mj': round(energy_mj, 3),
        'infs_per_wh': round(infs_per_wh, 0),
        'cpu_pct': round(cpu_pct, 1),
    }
    energy_results.append(row)
    print(f'  {tier_name:<26}  {profile["threads"]:>8}  {profile["power_w"]:>10.1f}  '
          f'{mean_ms:>10.2f}  {energy_mj:>12.3f}  {infs_per_wh:>10.0f}')

# Battery life calculation
print()
print('Battery Life Estimate (25 Wh drone battery @ 10 inferences/second):')
for r in energy_results:
    inf_power_w = r['energy_mj'] / 1000 * 10  # 10 inferences/sec
    flight_min = (25 * 3600) / (inf_power_w * 3600) / 60
    print(f'  {r["tier"]:<26}: inference consumes {inf_power_w:.2f} W -> battery supports {flight_min:.0f} min')

# Bar chart
os.makedirs('results', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('SkyRoute Drone - Energy Per Inference by Hardware Tier', fontsize=12, fontweight='bold')
tiers = [r['tier'] for r in energy_results]
energies = [r['energy_mj'] for r in energy_results]
infs_wh = [r['infs_per_wh'] for r in energy_results]
colors = ['#1C7293', '#02C39A', '#E07B39']

bars1 = axes[0].bar(tiers, energies, color=colors, edgecolor='white', alpha=0.85)
axes[0].set_ylabel('Energy per Inference (mJ)', fontsize=10)
axes[0].set_title('Lower is better for battery life')
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars1, energies):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(tiers, infs_wh, color=colors, edgecolor='white', alpha=0.85)
axes[1].set_ylabel('Inferences per Watt-hour', fontsize=10)
axes[1].set_title('Higher is better for efficiency')
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars2, infs_wh):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/energy_analysis.png', dpi=120, bbox_inches='tight')
plt.show()

with open('results/energy.json', 'w') as f:
    json.dump(energy_results, f, indent=2)
print('\n✔ Chart saved to results/energy_analysis.png')