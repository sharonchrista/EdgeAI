"""
step1_train_model.py
Lab 3 - Step 1: Train a 3-class anomaly classifier and convert to TFLite.
Features: [temperature_celsius, vibration_amplitude]
Classes : 0=Normal  1=Warning  2=Critical
"""
import numpy as np
import tensorflow as tf
import os

np.random.seed(42)
tf.random.set_seed(42)

# 1. Generate synthetic training data
n_normal = 6000
temp_n = np.random.uniform(2.0, 8.0, n_normal)
vib_n = np.abs(np.random.normal(0.5, 0.3, n_normal))
X_normal = np.stack([temp_n, vib_n], axis=1)
y_normal = np.zeros(n_normal, dtype=np.int32)

n_warn = 2500
temp_w = np.random.uniform(8.0, 12.0, n_warn)
vib_w = np.random.uniform(1.5, 3.0, n_warn)
X_warn = np.stack([temp_w, vib_w], axis=1)
y_warn = np.ones(n_warn, dtype=np.int32)

n_crit = 1500
temp_c = np.random.uniform(12.0, 20.0, n_crit)
vib_c = np.random.uniform(3.0, 6.0, n_crit)
X_crit = np.stack([temp_c, vib_c], axis=1)
y_crit = np.full(n_crit, 2, dtype=np.int32)

X = np.vstack([X_normal, X_warn, X_crit]).astype(np.float32)
y = np.concatenate([y_normal, y_warn, y_crit])

# 2. Z-score normalisation -- CRITICAL: subscriber must reuse these exact stats
MEAN = X.mean(axis=0)
STD = X.std(axis=0)
X_norm = (X - MEAN) / STD
np.save('normalisation_stats.npy', np.array([MEAN, STD]))
print(f'Normalisation stats saved:')
print(f'  MEAN = {MEAN}')
print(f'  STD  = {STD}')

# 3. Shuffle and split
idx = np.random.permutation(len(X_norm))
X_norm, y = X_norm[idx], y[idx]
split = int(0.8 * len(X_norm))
X_train, X_val = X_norm[:split], X_norm[split:]
y_train, y_val = y[:split], y[split:]
print(f'Train: {len(X_train)}  |  Val: {len(X_val)}')
print(f'Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}')

# 4. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
], name='safevax_anomaly_detector')
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nValidation accuracy : {val_acc:.4f}  ({val_acc*100:.1f}%)')

# 6. Sanity checks
test_inputs = np.array([
    [5.0, 0.4],
    [10.0, 2.0],
    [15.0, 4.5],
], dtype=np.float32)
test_norm = (test_inputs - MEAN) / STD
preds = model.predict(test_norm, verbose=0).argmax(axis=1)
labels = ['Normal', 'Warning', 'Critical']
for inp, pred in zip(test_inputs, preds):
    print(f'  temp={inp[0]:.1f}C  vib={inp[1]:.1f}AU  ->  {labels[pred]}')

# 7. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('anomaly_model.tflite', 'wb') as f:
    f.write(tflite_model)
size_kb = os.path.getsize('anomaly_model.tflite') / 1024
print(f'\n✔ anomaly_model.tflite saved  ({size_kb:.1f} KB)')
print('✔ normalisation_stats.npy saved')
print('Both files must be present before running the subscriber.')