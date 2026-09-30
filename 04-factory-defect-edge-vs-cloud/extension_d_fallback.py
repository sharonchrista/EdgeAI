"""
extension_d_fallback.py
Lab 1 - Extension D: Connectivity Failure Simulation.
Implements retry-with-local-fallback: if the cloud server is unreachable
within a timeout, fall back to local TFLite inference and log the decision.
Tests both the failure path (server down) and success path (server up).
"""
import time
import json
import os
import logging
import numpy as np
import requests
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- Logging setup: every decision (cloud vs fallback) gets recorded ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('results/connectivity_decisions.log', mode='a'),
        logging.StreamHandler()
    ]
)
os.makedirs('results', exist_ok=True)

URL = "http://127.0.0.1:8000/predict"
CLOUD_TIMEOUT_S = 1.5   # how long we wait for the cloud before giving up
MAX_RETRIES = 2         # retry attempts before falling back to local inference

MODEL_PATH = 'model.tflite'
x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \
    else np.random.rand(1, 28, 28, 1).astype(np.float32)
payload = {"pixels": x.flatten().tolist()}

# --- Local TFLite interpreter, loaded once and kept ready as the fallback ---
_interp = Interpreter(model_path=MODEL_PATH)
_interp.allocate_tensors()
_inp = _interp.get_input_details()[0]
_out = _interp.get_output_details()[0]


def local_inference():
    """Run inference entirely on-device. This is the fallback path."""
    t0 = time.perf_counter()
    _interp.set_tensor(_inp['index'], x)
    _interp.invoke()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    output = _interp.get_tensor(_out['index'])
    pred = int(np.argmax(output))
    conf = float(np.max(output))
    return pred, conf, elapsed_ms


def try_cloud_with_retry():
    """
    Attempt cloud inference with retries. Returns (success, result_dict).
    On any failure (timeout, connection error) after all retries are
    exhausted, returns success=False so the caller can fall back locally.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            r = requests.post(URL, json=payload, timeout=CLOUD_TIMEOUT_S)
            r.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = r.json()
            logging.info(f'CLOUD SUCCESS  (attempt {attempt}/{MAX_RETRIES})  '
                         f'latency={elapsed_ms:.2f}ms  pred={result["prediction"]}')
            return True, {'source': 'cloud', 'attempt': attempt,
                          'latency_ms': elapsed_ms, **result}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logging.warning(f'CLOUD ATTEMPT {attempt}/{MAX_RETRIES} FAILED  '
                            f'({type(e).__name__})')
            if attempt < MAX_RETRIES:
                time.sleep(0.3)  # brief backoff before retry
    return False, None


def infer_with_fallback():
    """
    The main decision function a production edge device would call.
    Tries cloud first (with retries); falls back to local TFLite inference
    if the cloud is unreachable. Every decision is logged.
    """
    success, cloud_result = try_cloud_with_retry()
    if success:
        return cloud_result

    logging.warning('CLOUD UNREACHABLE after all retries -- '
                    'FALLING BACK TO LOCAL INFERENCE')
    pred, conf, elapsed_ms = local_inference()
    logging.info(f'LOCAL FALLBACK SUCCESS  latency={elapsed_ms:.3f}ms  '
                f'pred={pred}  confidence={conf:.4f}')
    return {'source': 'local_fallback', 'prediction': pred,
            'confidence': conf, 'latency_ms': elapsed_ms}


if __name__ == '__main__':
    print('=' * 65)
    print('  CONNECTIVITY FAILURE SIMULATION')
    print(f'  Cloud timeout: {CLOUD_TIMEOUT_S}s  |  Max retries: {MAX_RETRIES}')
    print('  Decisions logged to results/connectivity_decisions.log')
    print('=' * 65)

    print('\nRunning inference (checks cloud availability automatically)...\n')
    result = infer_with_fallback()

    print('\n' + '-' * 65)
    print('  RESULT')
    print('-' * 65)
    print(f'  Source       : {result["source"]}')
    print(f'  Prediction   : {result["prediction"]}')
    print(f'  Confidence   : {result["confidence"]:.4f}')
    print(f'  Latency      : {result["latency_ms"]:.3f} ms')
    print('-' * 65)

    if result['source'] == 'local_fallback':
        print('\n  >> System degraded gracefully: cloud was unreachable,')
        print('     but the drone/camera/device still got a prediction.')
    else:
        print('\n  >> Cloud path succeeded normally.')