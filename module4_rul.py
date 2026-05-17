import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from nvh_core import GPUTrainer, get_device, BaseModel

# ---------------------------------------------------------
# Data Loading and Piecewise RUL Calculation
# ---------------------------------------------------------
def prepare_cmapss_data(train_path, test_path, seq_len=30):
    # Simulated columns for NASA CMAPSS dataset
    cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    
    # Generate dummy data if files don't exist
    if not os.path.exists(train_path):
        print(f"WARNING: {train_path} not found. Generating dummy CMAPSS data.")
        df_train = pd.DataFrame(np.random.rand(5000, 26), columns=cols)
        df_train['engine_id'] = np.repeat(np.arange(1, 26), 200)
        df_train['cycle'] = np.tile(np.arange(1, 201), 25)
    else:
        df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
        
    # We only keep 14 critical sensors (standard for FD001)
    sensors_to_keep = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # Calculate RUL
    rul = pd.DataFrame(df_train.groupby('engine_id')['cycle'].max()).reset_index()
    rul.columns = ['engine_id', 'max_cycle']
    df_train = df_train.merge(rul, on=['engine_id'], how='left')
    df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
    # Piecewise linear: cap RUL at 125
    df_train['RUL'] = df_train['RUL'].apply(lambda x: min(125, x))
    df_train.drop('max_cycle', axis=1, inplace=True)
    
    # Scale features
    scaler = MinMaxScaler()
    df_train[sensors_to_keep] = scaler.fit_transform(df_train[sensors_to_keep])
    
    # Generate Sequences
    def gen_sequence(id_df, seq_length, seq_cols):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]
            
    def gen_labels(id_df, seq_length, label):
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]

    seq_gen = (list(gen_sequence(df_train[df_train['engine_id'] == id], seq_len, sensors_to_keep)) 
               for id in df_train['engine_id'].unique())
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    label_gen = [gen_labels(df_train[df_train['engine_id'] == id], seq_len, ['RUL']) 
                 for id in df_train['engine_id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)
    
    return seq_array, label_array

# ---------------------------------------------------------
# NASA Scoring Function
# ---------------------------------------------------------
def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = 0
    for error in d:
        if error < 0:
            score += np.exp(-error / 13) - 1
        else:
            score += np.exp(error / 10) - 1
    return score

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class RUL_LSTM(BaseModel):
    def __init__(self):
        super(RUL_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last cycle
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target):
        return self.mse(pred, target) + 0.1 * self.mae(pred, target)

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    device = get_device()
    train_path = 'Module4_RUL/train_FD001.txt'
    test_path = 'Module4_RUL/test_FD001.txt'
    
    X, y = prepare_cmapss_data(train_path, test_path, seq_len=30)
    print(f"RUL Dataset shape: X={X.shape}, y={y.shape}")
    
    # 80/20 split for demo (Since no explicit test logic provided without actual test_FD001 layout)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False, pin_memory=True)
    
    print("\nTraining RUL LSTM Model...")
    model = RUL_LSTM()
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    trainer = GPUTrainer(model, train_loader, val_loader, criterion, optimizer, 
                         epochs=150, patience=20, scheduler=scheduler, device=device)
    model = trainer.train()
    
    torch.save(model.state_dict(), 'models/module4_rul_lstm.pt')
    
    # Evaluation
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch in DataLoader(X_val_t, batch_size=64):
            preds = model(X_batch.to(device)).cpu().numpy()
            y_pred.extend(preds)
            
    y_pred = np.array(y_pred).flatten()
    y_true = y_val_t.numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)
    
    print("\n--- RUL Predictor Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"NASA Scoring Function: {score:.2f}")
    
    # Plotting first 100 validation engines (simulated as 5 distinct run-to-failure curves)
    plt.figure(figsize=(10,5))
    plt.plot(y_true[:100], label='Actual RUL', color='blue')
    plt.plot(y_pred[:100], label='Predicted RUL', color='red', linestyle='dashed')
    plt.title("RUL Prediction vs Actual for Test Engines")
    plt.xlabel('Time (Cycles)')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/Module4_RUL_plot.png')
    plt.close()
    
if __name__ == "__main__":
    main()
