import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet18
import warnings
import cv2

warnings.filterwarnings('ignore')

from nvh_core import SignalLoader, Preprocessor, get_device, BaseModel

# ---------------------------------------------------------
# Order Tracking (Heuristic RPM Harmonics)
# ---------------------------------------------------------
def apply_order_tracking(signal, base_rpm=3000, sr=12000, num_harmonics=5):
    """
    Simulates order tracking by extracting synchronous harmonics.
    In a real scenario, this uses tachometer signals to resample 
    from the time domain to the angular domain.
    Here we apply a bandpass filter around the expected harmonics.
    """
    from scipy.signal import butter, filtfilt
    
    base_hz = base_rpm / 60.0
    filtered_signal = np.zeros_like(signal)
    
    # Extract frequencies around 1X, 2X, 3X... order harmonics
    nyq = 0.5 * sr
    for i in range(1, num_harmonics + 1):
        center_freq = base_hz * i
        low = max(0.1, (center_freq - 10)) / nyq
        high = min(0.99, (center_freq + 10)) / nyq
        
        b, a = butter(4, [low, high], btype='band')
        harmonic_sig = filtfilt(b, a, signal)
        filtered_signal += harmonic_sig
        
    return filtered_signal

# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------
def load_ev_motor_dataset(data_dir):
    print("Loading EV Motor Dataset...")
    # Simulated mapping: Normal=0, ElectricalFault=1, MechanicalFault=2
    # Assuming CSV or MAT files exist
    files = glob.glob(os.path.join(data_dir, '*.*'))
    
    X_spec = []
    y = []
    
    if len(files) == 0:
        print("WARNING: No data files found in", data_dir)
        print("Generating synthetic dummy data for demonstration...")
        for label in range(3):
            for _ in range(120):
                # Simulated 12000 points (1 second)
                sig = np.random.randn(12000) + label * 0.4 * np.sin(np.linspace(0, 100, 12000))
                tracked_sig = apply_order_tracking(sig)
                spec = Preprocessor.stft_spectrogram(tracked_sig)
                X_spec.append([spec]) # Add channel dimension
                y.append(label)
    else:
        for file in files:
            file_lower = file.lower()
            label = 0
            if 'elec' in file_lower: label = 1
            elif 'mech' in file_lower: label = 2
            
            sig = SignalLoader.load_csv(file) if file.endswith('.csv') else SignalLoader.load_mat(file)
            tracked_sig = apply_order_tracking(sig)
            spec = Preprocessor.stft_spectrogram(tracked_sig)
            X_spec.append([spec])
            y.append(label)

    return np.array(X_spec), np.array(y)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class EVResNet18(BaseModel):
    def __init__(self):
        super(EVResNet18, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 3 classes: Normal, Electrical, Mechanical
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# Custom Trainer for Gradient Accumulation & OneCycleLR
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, device, epochs=60, accum_steps=2):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    
    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader) // accum_steps + 1, epochs=epochs
    )
    
    scaler = GradScaler()
    best_loss = float('inf')
    best_model_wts = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            with autocast():
                outputs = model(X)
                loss = criterion(outputs, y)
                # Normalize loss to account for accumulation
                loss = loss / accum_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            train_loss += loss.item() * accum_steps
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast():
                    outputs = model(X)
                    loss = criterion(outputs, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | VRAM: {torch.cuda.memory_allocated(device)/(1024**2):.1f} MB")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            
        torch.cuda.empty_cache()
        
    model.load_state_dict(best_model_wts)
    return model

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    device = get_device()
    data_dir = 'Module3_EV_Motor'
    
    X_spec, y = load_ev_motor_dataset(data_dir)
    print(f"Dataset shape: X={X_spec.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, stratify=y_train, random_state=42)
    
    print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # -----------------------------------------------------
    # Dataloaders
    # -----------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
    
    print("\nTraining EV ResNet-18 with Gradient Accumulation (accum_steps=2)...")
    model = EVResNet18()
    model = train_model(model, train_loader, val_loader, device, epochs=60, accum_steps=2)
    
    torch.save(model.state_dict(), 'models/module3_ev_resnet18.pt')
    
    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            
    print("\n--- EV Motor Classification Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    
    target_names = ['Normal', 'ElectricalFault', 'MechanicalFault']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("EV Motor - Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('logs/Module3_EV_cm.png')
    plt.close()

if __name__ == "__main__":
    main()
