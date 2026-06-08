import json
import os
import pickle

import numpy as np

model = None


def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)


def run(raw_data):
    payload = json.loads(raw_data)

    if "input_data" in payload:
        data = payload["input_data"]["data"]
    elif "data" in payload:
        data = payload["data"]
    else:
        raise ValueError("Expected 'data' or 'input_data' in request body.")

    predictions = model.predict(np.array(data))
    return {"predictions": predictions.tolist()}
