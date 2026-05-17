from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import json
import time
import uvicorn
import numpy as np
import datetime

from nvh_core import ModuleRouter
from database import init_db, get_db, DiagnosticRecord

app = FastAPI(title="NVH AI System API")

# Initialize DB tables on startup
@app.on_event("startup")
def on_startup():
    init_db()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = ModuleRouter()

def save_diagnostic_to_db(db: Session, res: dict):
    db_record = DiagnosticRecord(
        signal_id=res["signal_id"],
        timestamp=datetime.datetime.strptime(res["timestamp"], '%Y-%m-%d %H:%M:%S'),
        module_used=res["module_used"],
        fault_detected=res["fault_detected"],
        fault_type=res["fault_type"],
        severity_score=res["severity_score"],
        rul_cycles=res["rul_cycles"],
        maintenance_alert=res["maintenance_alert"],
        confidence=res["confidence"],
        processing_time_ms=res["processing_time_ms"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Simulate reading file
    contents = await file.read()
    # Mocking signal array
    signal = np.random.randn(2000) 
    result = json.loads(router.analyze_signal(signal, signal_id=file.filename))
    
    # Save to PostgreSQL
    save_diagnostic_to_db(db, result)
    
    return result

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    results = []
    for file in files:
        signal = np.random.randn(2000)
        res = json.loads(router.analyze_signal(signal, signal_id=file.filename))
        save_diagnostic_to_db(db, res)
        results.append(res)
    return results

@app.get("/history")
async def get_history(limit: int = 100, db: Session = Depends(get_db)):
    records = db.query(DiagnosticRecord).order_by(DiagnosticRecord.timestamp.desc()).limit(limit).all()
    return records

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
