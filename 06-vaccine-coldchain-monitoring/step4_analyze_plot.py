"""
step4_analyze_plot.py
Lab 3 - Step 4: Query the SQLite log and visualise the full run as a
time-series dashboard -- temperature, vibration, predicted class, and
alert points, all on a shared timeline.
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

DB_PATH = 'edge_data.db'

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f'{DB_PATH} not found. Run step3_edge_subscriber.py first '
        f'(with step2_sensor_publisher.py feeding it) to generate data.'
    )

# 1. Query all logged inferences
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query('SELECT * FROM inferences ORDER BY timestamp ASC', conn)
conn.close()

print(f'Loaded {len(df)} logged inferences from {DB_PATH}')
print(f'\nClass distribution:')
print(df['predicted_class'].value_counts().to_string())
print(f'\nTotal alerts raised: {df["is_alert"].sum()}')

if len(df) == 0:
    raise ValueError('Database is empty -- no inferences were logged.')

# 2. Convert timestamp to elapsed seconds and readable datetime
df['elapsed_s'] = df['timestamp'] - df['timestamp'].iloc[0]
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# 3. Class -> numeric mapping for plotting
class_to_num = {'Normal': 0, 'Warning': 1, 'Critical': 2}
df['class_num'] = df['predicted_class'].map(class_to_num)

class_colors = {'Normal': '#02C39A', 'Warning': '#F4A300', 'Critical': '#E63946'}
df['color'] = df['predicted_class'].map(class_colors)

# 4. Build the dashboard: 3 stacked panels sharing the same x-axis
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle('SafeVax Cold-Chain Monitoring -- Edge Inference Timeline',
             fontsize=13, fontweight='bold')

# Panel 1: Temperature over time
axes[0].plot(df['elapsed_s'], df['temperature_c'], color='#1C7293', linewidth=1.2)
axes[0].scatter(df['elapsed_s'], df['temperature_c'], c=df['color'], s=15, zorder=5)
axes[0].axhline(10, color='orange', linestyle=':', linewidth=1, label='Warning threshold (~10C)')
axes[0].axhline(12, color='red', linestyle=':', linewidth=1, label='Critical threshold (~12C)')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Temperature Sensor')
axes[0].legend(fontsize=8, loc='upper left')
axes[0].grid(alpha=0.3)

# Panel 2: Vibration over time
axes[1].plot(df['elapsed_s'], df['vibration_au'], color='#065A82', linewidth=1.2)
axes[1].scatter(df['elapsed_s'], df['vibration_au'], c=df['color'], s=15, zorder=5)
axes[1].axhline(1.5, color='orange', linestyle=':', linewidth=1, label='Warning threshold (~1.5 AU)')
axes[1].axhline(3.0, color='red', linestyle=':', linewidth=1, label='Critical threshold (~3.0 AU)')
axes[1].set_ylabel('Vibration (AU)')
axes[1].set_title('Vibration Sensor (Compressor Motor)')
axes[1].legend(fontsize=8, loc='upper left')
axes[1].grid(alpha=0.3)

# Panel 3: Predicted class over time + alert markers
axes[2].scatter(df['elapsed_s'], df['class_num'], c=df['color'], s=20, zorder=5)
axes[2].plot(df['elapsed_s'], df['class_num'], color='gray', linewidth=0.6, alpha=0.5, zorder=1)
alert_rows = df[df['is_alert'] == 1]
if len(alert_rows) > 0:
    axes[2].scatter(alert_rows['elapsed_s'], alert_rows['class_num'],
                    marker='*', s=200, color='red', edgecolor='black',
                    linewidth=0.5, zorder=10, label=f'Alert fired (n={len(alert_rows)})')
    first_alert_s = alert_rows['elapsed_s'].iloc[0]
    axes[2].axvline(first_alert_s, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
    axes[2].annotate(f'First alert\nt={first_alert_s:.1f}s',
                     (first_alert_s, 2.1), fontsize=8, color='red', fontweight='bold')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['Normal', 'Warning', 'Critical'])
axes[2].set_ylabel('Predicted Class')
axes[2].set_xlabel('Elapsed Time (seconds)')
axes[2].set_title('Model Prediction & Alert Timeline')
axes[2].legend(fontsize=8, loc='upper left')
axes[2].grid(alpha=0.3)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/coldchain_timeline.png', dpi=120, bbox_inches='tight')
plt.show()
print('\n✔ Dashboard saved to results/coldchain_timeline.png')

# 5. Summary statistics for the report
print('\n' + '=' * 60)
print('  RUN SUMMARY')
print('=' * 60)
print(f'  Total inferences logged    : {len(df)}')
print(f'  Run duration                : {df["elapsed_s"].max():.1f}s')
print(f'  Normal readings             : {(df["predicted_class"]=="Normal").sum()}')
print(f'  Warning readings            : {(df["predicted_class"]=="Warning").sum()}')
print(f'  Critical readings           : {(df["predicted_class"]=="Critical").sum()}')
print(f'  Total alerts raised         : {df["is_alert"].sum()}')
if len(alert_rows) > 0:
    print(f'  Time to first alert         : {first_alert_s:.1f}s into the run')
    print(f'  (Recall: real vibration-based prediction window is 8-12 minutes')
    print(f'   before temperature crosses excursion threshold -- this simulated')
    print(f'   run compresses that into a {df["elapsed_s"].max():.0f}-second demonstration)')
print('=' * 60)