"""
convert_to_tflite.py
Lab 1 - Step 2: Convert trained model to TFLite with quantization.
"""
import tensorflow as tf
import os

# Load the saved Keras model
model = tf.keras.models.load_model('mnist_model.keras')
print('Loaded model from  mnist_model.keras')

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable dynamic-range quantization:
#   - Weights: FP32 -> INT8  (~4x size reduction)
#   - Activations: remain FP32 (no representative dataset needed)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Run conversion
tflite_model = converter.convert()

# Save the FlatBuffer
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Report compression
sm_size = os.path.getsize('mnist_model.keras') / 1024
tfl_size = os.path.getsize('model.tflite') / 1024
print(f'\n✔ Conversion complete!')
print(f'   SavedModel size  : {sm_size:,.1f} KB')
print(f'   TFLite size      : {tfl_size:,.1f} KB')
print(f'   Compression      : {sm_size/tfl_size:.1f}x')