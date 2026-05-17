import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet18
import warnings

warnings.filterwarnings('ignore')

from nvh_core import SignalLoader, Preprocessor, GPUTrainer, get_device, BaseModel

# ---------------------------------------------------------
# Dataset Loading & Segmentation
# ---------------------------------------------------------
def segment_signal(signal, window=1024, stride=512):
    segments = []
    for start in range(0, len(signal) - window + 1, stride):
        segments.append(signal[start:start + window])
    return np.array(segments)

def load_cwru_dataset(data_dir):
    print("Loading CWRU Dataset...")
    X_raw = []
    y = []
    
    # Simulating data loading mapping for the classes
    # If exact files are present, they are mapped here. 
    # For robust demonstration, we will generate synthetic data if files are not found
    mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
    
    if len(mat_files) == 0:
        print("WARNING: No .mat files found in", data_dir)
        print("Generating synthetic dummy data for demonstration...")
        # 0: Normal, 1: InnerRace, 2: Ball, 3: OuterRace
        for class_idx in range(4):
            # Generate 200 segments per class for demo
            for _ in range(200):
                sig = np.random.randn(1024) + class_idx * 0.5 * np.sin(np.linspace(0, 10, 1024))
                X_raw.append(sig)
                y.append(class_idx)
    else:
        # Actual loading logic (adjust keys based on actual CWRU structure)
        # Normal=0, InnerRace=1, Ball=2, OuterRace=3
        # Here we assume file names contain class info like 'normal', 'inner', etc.
        for file in mat_files:
            file_lower = file.lower()
            if 'normal' in file_lower: label = 0
            elif 'inner' in file_lower: label = 1
            elif 'ball' in file_lower: label = 2
            elif 'outer' in file_lower: label = 3
            else: continue
            
            sig = SignalLoader.load_mat(file)
            # Find DE_time key heuristically if needed, SignalLoader handles it
            segments = segment_signal(sig, window=1024, stride=512)
            X_raw.extend(segments)
            y.extend([label] * len(segments))
            
    return np.array(X_raw), np.array(y)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class CNN1D(BaseModel):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 64, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x # CrossEntropyLoss applies Softmax internally

class ResNet18_2D(BaseModel):
    def __init__(self):
        super(ResNet18_2D, self).__init__()
        self.model = resnet18(pretrained=False)
        # Modify first layer for 1-channel (grayscale) input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify last layer for 4 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# Evaluation & Reporting
# ---------------------------------------------------------
def evaluate_and_report(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Inner', 'Ball', 'Outer'],
                yticklabels=['Normal', 'Inner', 'Ball', 'Outer'])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'logs/{model_name.replace(" ", "_")}_cm.png')
    plt.close()
    print(f"Saved Confusion Matrix to logs/{model_name.replace(' ', '_')}_cm.png")

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    device = get_device()
    data_dir = 'Module1_Bearing'
    
    X_raw, y = load_cwru_dataset(data_dir)
    print(f"Dataset shape: X={X_raw.shape}, y={y.shape}")
    
    # Train/Val/Test Split (70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_raw, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42) # ~15% of total
    
    print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # -----------------------------------------------------
    # Model A: SVM Baseline (CPU)
    # -----------------------------------------------------
    print("\nTraining Model A - SVM Baseline...")
    # Extract features for SVM
    X_train_svm = np.array([list(Preprocessor.extract_features(sig).values()) for sig in X_train])
    X_test_svm = np.array([list(Preprocessor.extract_features(sig).values()) for sig in X_test])
    
    svm_model = SVC(kernel='rbf', C=10, gamma='scale')
    svm_model.fit(X_train_svm, y_train)
    y_pred_svm = svm_model.predict(X_test_svm)
    evaluate_and_report(y_test, y_pred_svm, "SVM Baseline")
    
    # -----------------------------------------------------
    # Model B: 1D-CNN
    # -----------------------------------------------------
    print("\nPreparing dataloaders for DL models...")
    # Add channel dimension (N, 1, 1024)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader_1d = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True, pin_memory=True)
    val_loader_1d = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False, pin_memory=True)
    
    print("Training Model B - 1D CNN...")
    model_b = CNN1D()
    criterion = nn.CrossEntropyLoss()
    optimizer_b = optim.Adam(model_b.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=50)
    
    trainer_b = GPUTrainer(model_b, train_loader_1d, val_loader_1d, criterion, optimizer_b, 
                           epochs=50, patience=10, scheduler=scheduler_b, device=device)
    model_b = trainer_b.train()
    
    # Save Model B
    torch.save(model_b.state_dict(), 'models/module1_cnn1d.pt')
    
    # Test Model B
    model_b.eval()
    y_pred_b = []
    with torch.no_grad():
        for X_batch in DataLoader(X_test_t, batch_size=32):
            preds = model_b(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            y_pred_b.extend(preds)
    evaluate_and_report(y_test, y_pred_b, "1D CNN")
    torch.cuda.empty_cache()
    
    # -----------------------------------------------------
    # Model C: 2D-CNN (STFT Spectrograms)
    # -----------------------------------------------------
    print("\nGenerating STFT Spectrograms for Model C...")
    # Generate images: (N, 1, 128, 128)
    def to_spectrograms(signals):
        return np.array([[Preprocessor.stft_spectrogram(sig)] for sig in signals])
        
    X_train_spec = torch.tensor(to_spectrograms(X_train), dtype=torch.float32)
    X_val_spec = torch.tensor(to_spectrograms(X_val), dtype=torch.float32)
    X_test_spec = torch.tensor(to_spectrograms(X_test), dtype=torch.float32)
    
    train_loader_2d = DataLoader(TensorDataset(X_train_spec, y_train_t), batch_size=32, shuffle=True, pin_memory=True)
    val_loader_2d = DataLoader(TensorDataset(X_val_spec, y_val_t), batch_size=32, shuffle=False, pin_memory=True)
    
    print("Training Model C - ResNet18 2D CNN...")
    model_c = ResNet18_2D()
    optimizer_c = optim.Adam(model_c.parameters(), lr=5e-4)
    
    trainer_c = GPUTrainer(model_c, train_loader_2d, val_loader_2d, criterion, optimizer_c, 
                           epochs=50, patience=10, scheduler=None, device=device)
    model_c = trainer_c.train()
    
    # Save Model C
    torch.save(model_c.state_dict(), 'models/module1_resnet18.pt')
    
    # Test Model C
    model_c.eval()
    y_pred_c = []
    with torch.no_grad():
        for X_batch in DataLoader(X_test_spec, batch_size=32):
            preds = model_c(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            y_pred_c.extend(preds)
    evaluate_and_report(y_test, y_pred_c, "ResNet18 2D CNN")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
