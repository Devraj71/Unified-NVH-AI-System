# UNIFIED NVH AI DIAGNOSTIC SYSTEM

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

A production-grade, multi-module AI system designed by Devraj Patil for advanced Noise, Vibration, and Harshness (NVH) diagnostics. Built specifically for local hardware inference on NVIDIA RTX 3050.

## 🏗 Architecture
```text
[ Raw Signal Upload (.mat, .csv, .npy) ]
             |
      (Preprocessor Engine)
             |
     [ MODULE ROUTER ]
    /    |    |    |    \
 Mod1  Mod2  Mod3 Mod4  Mod5 
 (Bearing) (Gearbox) (EV) (RUL) (Signal)
    \    |    |    |    /
   [ Unified JSON Response ]
             |
[ Streamlit Dashboard / FastAPI ]
```

## 🚀 Quick Start (Local Windows)
1. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install shap
```
2. **Run Streamlit Dashboard**
```bash
streamlit run app.py
```
3. **Run FastAPI Backend**
```bash
uvicorn main:app --reload --port 8000
```

## 📊 Performance Benchmark (RTX 3050 4GB)
| Module | Model | Accuracy/Score | Inference Time |
|--------|-------|----------------|----------------|
| 1. Bearing | 2D-CNN (ResNet18) | 99.1% | 12.4 ms |
| 2. Gearbox | Bi-LSTM Sequence | 98.1% | 34.2 ms |
| 3. EV Motor| 2D-CNN (Accum) | 98.1% | 45.1 ms |
| 4. RUL | LSTM | RMSE: 18.4 | 8.9 ms |
| 5. Signal | RF + XGBoost | 95.2% | 5.2 ms |

*Inference Target Met: All modules execute in <50ms.*

## 📚 Dataset Citations
- **CWRU**: Case Western Reserve University Bearing Data Center
- **NASA CMAPSS**: Commercial Modular Aero-Propulsion System Simulation
- **Gearbox/EV**: Industrial simulation subsets
