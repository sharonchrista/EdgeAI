"""
build_model.py
Lab 1 - Step 1: Train a compact MNIST CNN.
Represents the cloud training phase of the Edge AI lifecycle.
"""
import tensorflow as tf
import numpy as np

# 1. Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f'Training samples : {len(x_train):,}')
print(f'Test samples     : {len(x_test):,}')

# 2. Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu',
                           padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
], name='mnist_cnn')
model.summary()

# 3. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train
model.fit(x_train, y_train, epochs=5, batch_size=64,
          validation_split=0.1, verbose=1)

# 5. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy : {acc:.4f}  ({acc*100:.2f}%)')

# 6. Save
model.save('mnist_model.keras')
print('\n Model saved to  mnist_model.keras')

# Save one test sample for consistent benchmarking later
np.save('test_sample.npy', x_test[:1])
print(f'Test sample saved  (true label: {y_test[0]})')