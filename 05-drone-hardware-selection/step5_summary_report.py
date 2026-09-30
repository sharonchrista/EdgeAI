"""
step5_summary_report.py
Lab 2 - Step 5: Aggregate all benchmark results into a unified comparison
and generate the hardware recommendation for SkyRoute drone.
"""
import json, os, statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_json(path, fallback):
    try:
        return json.load(open(path))
    except Exception:
        print(f'WARNING: {path} not found. Run the corresponding step first.')
        return fallback


latency_data = load_json('results/latency_sweep.json', [])
throughput_data = load_json('results/throughput.json', [])
energy_data = load_json('results/energy.json', [])

print('=' * 80)
print('  HARDWARE BENCHMARK SUMMARY - SkyRoute Drone Hardware Selection')
print('=' * 80)

if latency_data:
    print(f'\n  LATENCY (200 runs, 10 warmup, MobileNetV2 INT8):')
    print(f'  {"Threads":>8}  {"Mean ms":>10}  {"Median ms":>10}  {"p95 ms":>9}  {"p99 ms":>9}')
    print('  ' + '-' * 55)
    for r in latency_data:
        print(f'  {r["threads"]:>8}  {r["mean_ms"]:>10.2f}  {r["median_ms"]:>10.2f}  '
              f'{r["p95_ms"]:>9.2f}  {r["p99_ms"]:>9.2f}')

if throughput_data:
    print(f'\n  CONCURRENT THROUGHPUT:')
    print(f'  {"Workers":>8}  {"Throughput(ips)":>16}  {"Mean lat(ms)":>13}  {"p95 lat(ms)":>12}')
    print('  ' + '-' * 55)
    for r in throughput_data:
        print(f'  {r["workers"]:>8}  {r["throughput_ips"]:>16.2f}  '
              f'{r["mean_lat_ms"]:>13.2f}  {r["p95_lat_ms"]:>12.2f}')

if energy_data:
    print(f'\n  ENERGY PER INFERENCE:')
    print(f'  {"Tier":<26}  {"Power(W)":>9}  {"Latency(ms)":>12}  {"Energy(mJ)":>11}  {"inf/Wh":>8}')
    print('  ' + '-' * 73)
    for r in energy_data:
        print(f'  {r["tier"]:<26}  {r["power_w"]:>9.1f}  {r["mean_ms"]:>12.2f}  '
              f'{r["energy_mj"]:>11.3f}  {r["infs_per_wh"]:>8.0f}')

# Visualisation dashboard
os.makedirs('results', exist_ok=True)
if latency_data and energy_data:
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('SkyRoute Drone - Hardware Benchmark Dashboard (Lab 2)',
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.32)
    colors = ['#1C7293', '#02C39A', '#E07B39', '#065A82']

    # Panel 1: Latency vs thread count
    ax1 = fig.add_subplot(gs[0, 0])
    threads = [r['threads'] for r in latency_data]
    means = [r['mean_ms'] for r in latency_data]
    p95s = [r['p95_ms'] for r in latency_data]
    ax1.plot(threads, means, 'b-o', label='Mean latency', linewidth=2)
    ax1.plot(threads, p95s, 'r--o', label='p95 latency', linewidth=2)
    ax1.axhline(33.3, color='green', linestyle=':', linewidth=1.5,
                label='30 FPS budget (33.3 ms)')
    ax1.set_xlabel('TFLite Threads')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency vs Thread Count')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: Energy per inference
    ax2 = fig.add_subplot(gs[0, 1])
    tiers = [r['tier'].split('(')[0].strip() for r in energy_data]
    energies = [r['energy_mj'] for r in energy_data]
    bars = ax2.bar(tiers, energies, color=colors[:len(tiers)], edgecolor='white', alpha=0.85)
    ax2.set_ylabel('Energy per Inference (mJ)')
    ax2.set_title('Energy Efficiency by Tier')
    ax2.tick_params(axis='x', rotation=10)
    for bar, val in zip(bars, energies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: p95 latency bar
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar([str(t) for t in threads], p95s, color=colors[:len(threads)],
            edgecolor='white', alpha=0.85)
    ax3.axhline(33.3, color='red', linestyle='--', linewidth=1.5, label='30 FPS SLA')
    ax3.set_xlabel('Threads')
    ax3.set_ylabel('p95 Latency (ms)')
    ax3.set_title('p95 Latency - SLA Compliance')
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Inferences per Wh
    ax4 = fig.add_subplot(gs[1, 1])
    infs_wh = [r['infs_per_wh'] for r in energy_data]
    ax4.bar(tiers, infs_wh, color=colors[:len(tiers)], edgecolor='white', alpha=0.85)
    ax4.set_ylabel('Inferences per Watt-hour')
    ax4.set_title('Compute Efficiency (Higher = Better Battery)')
    ax4.tick_params(axis='x', rotation=10)
    ax4.grid(axis='y', alpha=0.3)

    plt.savefig('results/hardware_dashboard.png', dpi=120, bbox_inches='tight')
    plt.show()
    print('\n✔ Dashboard saved to results/hardware_dashboard.png')