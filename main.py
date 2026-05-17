from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import uvicorn
import numpy as np

from nvh_core import ModuleRouter

app = FastAPI(title="NVH AI System API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = ModuleRouter()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Simulate reading file
    contents = await file.read()
    # Mocking signal array
    signal = np.random.randn(2000) 
    result = router.analyze_signal(signal, signal_id=file.filename)
    return json.loads(result)

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        signal = np.random.randn(2000)
        res = json.loads(router.analyze_signal(signal, signal_id=file.filename))
        results.append(res)
    return results

@app.get("/health")
async def health():
    return {"status": "ok", "gpu_active": router.device.type == 'cuda'}

@app.get("/modules")
async def modules():
    return {
        "Module1": "Bearing (SVM, CNN1D, CNN2D)",
        "Module2": "Gearbox (XGB, LSTM)",
        "Module3": "EV Motor (ResNet18)",
        "Module4": "RUL (LSTM)",
        "Module5": "Signal Health (RF+XGB)"
    }

@app.get("/benchmark")
async def benchmark():
    router.benchmark_all_modules()
    return {"status": "Benchmark complete. View terminal logs."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
