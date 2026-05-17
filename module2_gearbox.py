import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cv2

warnings.filterwarnings('ignore')

from nvh_core import SignalLoader, Preprocessor, GPUTrainer, get_device, BaseModel

# ---------------------------------------------------------
# Dataset Loading & Preprocessing
# ---------------------------------------------------------
def load_gearbox_dataset(data_dir):
    print("Loading Gearbox .csv Dataset...")
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    X_raw = []
    y = []
    
    # Class mapping
    class_map = {'healthy': 0, 'brokentooth': 1, 'missingtooth': 2, 'crack': 3}
    
    if len(csv_files) == 0:
        print("WARNING: No .csv files found in", data_dir)
        print("Generating synthetic dummy data for demonstration...")
        for class_name, label in class_map.items():
            for _ in range(150):
                sig = np.random.randn(2000) + label * 0.3 * np.sin(np.linspace(0, 20, 2000))
                X_raw.append(sig)
                y.append(label)
    else:
        for file in csv_files:
            file_lower = file.lower()
            label = None
            for key, val in class_map.items():
                if key in file_lower:
                    label = val
                    break
            if label is None:
                continue
            
            sig = SignalLoader.load_csv(file)
            # Windowing large files into 2000 sample segments for consistency
            for start in range(0, len(sig) - 2000 + 1, 1000):
                X_raw.append(sig[start:start+2000])
                y.append(label)

    return np.array(X_raw), np.array(y)

# ---------------------------------------------------------
# Feature Extraction for LSTM and XGBoost
# ---------------------------------------------------------
def prepare_lstm_data(X_raw, seq_len=50):
    # LSTM expects (batch, seq_len, 8 features)
    X_seq = []
    for sig in X_raw:
        # Split signal into 'seq_len' chunks
        chunks = np.array_split(sig, seq_len)
        seq_features = []
        for chunk in chunks:
            feats = Preprocessor.extract_features(chunk)
            seq_features.append(list(feats.values()))
        X_seq.append(seq_features)
    return np.array(X_seq)

def prepare_xgb_data(X_raw):
    # For XGBoost: Compare Wavelet CWT vs FFT
    X_wavelet = []
    X_fft = []
    for sig in X_raw:
        # Wavelet CWT -> resize to smaller 1D representation to prevent memory bloat
        cwt_mat = Preprocessor.wavelet_cwt(sig, scales=32)
        cwt_resized = cv2.resize(np.abs(cwt_mat), (32, 32)).flatten()
        X_wavelet.append(cwt_resized)
        
        # FFT features
        _, amps, dom_freq = Preprocessor.fft_spectrum(sig)
        # Downsample FFT to 1024 bins for feature vector
        fft_downsampled = cv2.resize(amps.reshape(1, -1), (1024, 1)).flatten()
        X_fft.append(fft_downsampled)
        
    return np.array(X_wavelet), np.array(X_fft)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class LSTMModel(BaseModel):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=0.3, 
            batch_first=True, 
            bidirectional=True
        )
        # Bidirectional means hidden size is doubled
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        # Take the last time step from both directions
        out = out[:, -1, :] 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate_model(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC-AUC (OVR): {roc_auc:.4f}")
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Healthy', 'Broken', 'Missing', 'Crack'],
                yticklabels=['Healthy', 'Broken', 'Missing', 'Crack'])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'logs/{model_name.replace(" ", "_")}_cm.png')
    plt.close()

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    device = get_device()
    data_dir = 'Module2_Gearbox'
    
    X_raw, y = load_gearbox_dataset(data_dir)
    print(f"Dataset shape: X={X_raw.shape}, y={y.shape}")
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # -----------------------------------------------------
    # XGBoost: Wavelet CWT vs FFT
    # -----------------------------------------------------
    print("\nPreparing XGBoost Data (CWT vs FFT)...")
    X_train_cwt, X_train_fft = prepare_xgb_data(X_train)
    X_test_cwt, X_test_fft = prepare_xgb_data(X_test)
    
    print("Training XGBoost (Wavelet CWT Features)...")
    xgb_cwt = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric='mlogloss')
    xgb_cwt.fit(X_train_cwt, y_train)
    y_pred_cwt = xgb_cwt.predict(X_test_cwt)
    y_prob_cwt = xgb_cwt.predict_proba(X_test_cwt)
    evaluate_model(y_test, y_pred_cwt, y_prob_cwt, "XGBoost_Wavelet_CWT")
    
    print("Training XGBoost (FFT Features)...")
    xgb_fft = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric='mlogloss')
    xgb_fft.fit(X_train_fft, y_train)
    y_pred_fft = xgb_fft.predict(X_test_fft)
    y_prob_fft = xgb_fft.predict_proba(X_test_fft)
    evaluate_model(y_test, y_pred_fft, y_prob_fft, "XGBoost_FFT")
    
    # -----------------------------------------------------
    # LSTM: GPU Training
    # -----------------------------------------------------
    print("\nPreparing dataloaders for LSTM...")
    X_train_lstm = prepare_lstm_data(X_train, seq_len=50)
    X_val_lstm = prepare_lstm_data(X_val, seq_len=50)
    X_test_lstm = prepare_lstm_data(X_test, seq_len=50)
    
    X_train_t = torch.tensor(X_train_lstm, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_lstm, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test_lstm, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader_lstm = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True, pin_memory=True)
    val_loader_lstm = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False, pin_memory=True)
    
    print("Training LSTM Sequence Model...")
    model_lstm = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=1e-3)
    
    trainer_lstm = GPUTrainer(model_lstm, train_loader_lstm, val_loader_lstm, criterion, optimizer_lstm, 
                              epochs=100, patience=15, scheduler=None, device=device)
    model_lstm = trainer_lstm.train()
    
    # Save Model
    torch.save(model_lstm.state_dict(), 'models/module2_lstm.pt')
    
    # Test Model
    model_lstm.eval()
    y_pred_lstm = []
    y_prob_lstm = []
    with torch.no_grad():
        for X_batch in DataLoader(X_test_t, batch_size=64):
            X_batch = X_batch.to(device)
            out = model_lstm(X_batch)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred_lstm.extend(preds)
            y_prob_lstm.extend(probs)
            
    evaluate_model(y_test, y_pred_lstm, np.array(y_prob_lstm), "LSTM_Seq")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
