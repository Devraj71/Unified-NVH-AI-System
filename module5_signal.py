import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings('ignore')

from nvh_core import Preprocessor

# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------
def load_signal_health_dataset(data_dir):
    print("Loading Vibration Faults Dataset...")
    X_features = []
    y = []
    
    files = glob.glob(os.path.join(data_dir, '*.*'))
    if len(files) == 0:
        print("WARNING: No files found. Generating dummy dataset...")
        # Healthy=0, Faulty=1
        for label in [0, 1]:
            for _ in range(500):
                sig = np.random.randn(2000) + label * 0.8 * np.sin(np.linspace(0, 50, 2000))
                feats = Preprocessor.extract_features(sig)
                X_features.append(list(feats.values()))
                y.append(label)
    else:
        # User defined logic here. E.g., healthy strings mapped to 0
        pass 
        
    return np.array(X_features), np.array(y)

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    data_dir = 'Module5_Signal'
    X, y = load_signal_health_dataset(data_dir)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("\nTraining Ensemble (Random Forest + XGBoost)...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft', weights=[0.4, 0.6])
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Signal Health Classifier Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('logs/Module5_ROC.png')
    plt.close()
    
    # SHAP feature importance plot (using XGBoost base since SHAP handles it natively)
    if shap is not None:
        print("\nGenerating SHAP Summary Plot...")
        xgb.fit(X_train, y_train) # Re-fit solo to get explainer working cleanly
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)
        
        feature_names = ['RMS', 'Kurtosis', 'CrestFactor', 'Skewness', 'PeakToPeak', 'Variance', 'ZeroCrossingRate', 'SpectralEntropy']
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('logs/Module5_SHAP.png')
        plt.close()
    else:
        print("\nSkipping SHAP. Please `pip install shap` to enable feature importance.")

if __name__ == "__main__":
    main()
