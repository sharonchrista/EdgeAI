"""
cloud_server.py
Lab 1 - Step 4a: FastAPI inference server (cloud simulation).
"""
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI()

interpreter = Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]


class ImageRequest(BaseModel):
    pixels: list  # flattened 28*28*1 float list


@app.post("/predict")
def predict(req: ImageRequest):
    x = np.array(req.pixels, dtype=np.float32).reshape(1, 28, 28, 1)
    interpreter.set_tensor(inp['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(out['index'])
    pred = int(np.argmax(output))
    conf = float(np.max(output))
    return {"prediction": pred, "confidence": conf}


@app.get("/health")
def health():
    return {"status": "ok"}