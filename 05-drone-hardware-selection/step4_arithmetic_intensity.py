"""
step4_arithmetic_intensity.py
Lab 2 - Step 4: Arithmetic Intensity calculation and Roofline model sketch.
Computes FLOPs and memory bytes for MobileNetV2 and plots the Roofline.
"""
import numpy as np
import json, os, time, statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = 'model.tflite'

# 1. MobileNetV2 FLOPs and bytes (published values for 224x224 INT8)
# Source: Google MobileNetV2 paper + TFLite benchmark
MOBILENETV2_GFLOPS = 0.300   # 300 MFLOPs for 224x224 input
MOBILENETV2_FLOPS = MOBILENETV2_GFLOPS * 1e9
# Bytes transferred per inference (weights read + activation tensors)
# MobileNetV2 INT8: ~3.5 MB weights + ~5.5 MB peak activations = ~9 MB total
MOBILENETV2_BYTES = 9 * 1024 * 1024  # 9 MB
ARITHMETIC_INTENSITY = MOBILENETV2_FLOPS / MOBILENETV2_BYTES  # FLOP / byte

print('=== MobileNetV2 Arithmetic Intensity ===')
print(f'  FLOPs        : {MOBILENETV2_FLOPS/1e6:.0f} M  ({MOBILENETV2_GFLOPS:.3f} GFLOP)')
print(f'  Bytes moved  : {MOBILENETV2_BYTES/1024/1024:.1f} MB')
print(f'  AI (FLOP/B)  : {ARITHMETIC_INTENSITY:.1f}')

# 2. Hardware specs (approximate for each tier)
HARDWARE = {
    'MCU-tier\n(Cortex-M7 ~600MHz)': {
        'peak_gflops': 0.012,     # ~12 MFLOP/s (scalar FP32)
        'peak_bw_gb': 0.2,        # ~200 MB/s SRAM bandwidth
        'color': '#1C7293',
    },
    'Gateway-tier\n(RPi3 Cortex-A53)': {
        'peak_gflops': 13.5,      # ~13.5 GFLOP/s (NEON SIMD, 4 cores)
        'peak_bw_gb': 3.0,        # ~3 GB/s LPDDR2
        'color': '#02C39A',
    },
    'Edge-server\n(Jetson Orin NX)': {
        'peak_gflops': 1024.0,    # ~1 TFLOP INT8
        'peak_bw_gb': 68.0,       # ~68 GB/s LPDDR5
        'color': '#E07B39',
    },
}

# 3. Measure actual throughput (GFLOP/s achieved) on YOUR machine
print()
print('=== Achieved Throughput per Thread Configuration ===')
print(f'  {"Config":<30}  {"Mean lat (ms)":>14}  {"Achieved GFLOP/s":>17}')
print('  ' + '-' * 67)
achieved_pts = []
for threads, label in [(1, 'MCU-sim'), (4, 'Gateway-sim'), (8, 'EdgeServer-sim')]:
    interp = Interpreter(MODEL_PATH, num_threads=threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.rand(*tuple(inp['shape'])).astype(inp['dtype'])
    for _ in range(10):
        interp.set_tensor(inp['index'], x)
        interp.invoke()
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = statistics.mean(times)
    achieved = MOBILENETV2_FLOPS / (mean_ms / 1000) / 1e9
    print(f'  {label:<30}  {mean_ms:>14.2f}  {achieved:>17.3f}')
    achieved_pts.append({'label': label, 'ai': ARITHMETIC_INTENSITY, 'gflops': achieved, 'threads': threads})

# 4. Roofline diagram
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle('Roofline Model - MobileNetV2 INT8 on Edge Hardware Tiers',
             fontsize=12, fontweight='bold')
ai_range = np.logspace(-1, 4, 500)
for hw_name, hw in HARDWARE.items():
    mem_ceil = hw['peak_bw_gb'] * ai_range
    comp_ceil = np.full_like(ai_range, hw['peak_gflops'])
    roofline = np.minimum(mem_ceil, comp_ceil)
    ridge_ai = hw['peak_gflops'] / hw['peak_bw_gb']
    ax.loglog(ai_range, roofline, linewidth=2, color=hw['color'],
              label=f"{hw_name.replace(chr(10), ' ')}")
    ax.axvline(ridge_ai, color=hw['color'], linestyle=':', alpha=0.5, linewidth=1)

hw_colors = list(HARDWARE.values())
for i, pt in enumerate(achieved_pts):
    ax.scatter(pt['ai'], pt['gflops'], marker='*', s=250,
               color=hw_colors[i]['color'], zorder=5)
    ax.annotate(f'  {pt["label"]}\n  ({pt["gflops"]:.2f} GFLOP/s)',
                (pt['ai'], pt['gflops']), fontsize=8,
                color=hw_colors[i]['color'], fontweight='bold')

ax.axvline(ARITHMETIC_INTENSITY, color='red', linestyle='--', linewidth=1.5,
           label=f'MobileNetV2 AI = {ARITHMETIC_INTENSITY:.0f} FLOP/B')
ax.set_xlabel('Arithmetic Intensity (FLOP / Byte)', fontsize=11)
ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
ax.set_title('Stars = measured throughput | Curves = hardware ceilings')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('results/roofline.png', dpi=120, bbox_inches='tight')
plt.show()

print(f'\n✔ Roofline chart saved to results/roofline.png')
print(f'  Interpretation: is MobileNetV2 compute-bound or memory-bound on each tier?')
print(f"  Compare AI={ARITHMETIC_INTENSITY:.0f} against each tier's ridge point.")