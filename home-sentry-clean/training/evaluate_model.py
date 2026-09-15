import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

from home_sentry.model import WiFiShieldGRU
from home_sentry.features import SEQUENCE_LENGTH

# --- 2. CONFIGURATION ---
# Point this to your RAW, unscaled, unlabeled capture of the double attack
RAW_CSV = "Home_Data_Raw_2attack.csv" 
SCALER_PATH = "models/mega_both_scaler.pkl"
MODEL_PATH = "models/sentinel_shield_both_gru.pth"
OUTPUT_CSV = "ai_investigation_results1.csv"

def run_blind_investigation():
    print(f"🔍 Starting Blind Investigation on: {RAW_CSV}")

    # 1. Load Raw Data
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df):,} raw packets.")

    # Drop the Label column if it accidentally exists, otherwise proceed normally
    if 'Label' in df.columns:
        X_raw = df.drop(columns=['Label'])
    else:
        X_raw = df.copy()

    # 2. Scale the Data
    print(f"Loading Lens: {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)
    
    print("Translating raw packets...")
    # .transform() ignores the lack of labels and just applies the math
    X_scaled_array = scaler.transform(X_raw)

    # 3. Build Sequences Memory-Efficiently
    print("Building GRU packet sequences...")
    num_sequences = len(X_scaled_array) - SEQUENCE_LENGTH + 1
    
    # Pre-allocate numpy array for speed
    sequences = np.zeros((num_sequences, SEQUENCE_LENGTH, X_scaled_array.shape[1]), dtype=np.float32)
    for i in range(num_sequences):
        sequences[i] = X_scaled_array[i : i + SEQUENCE_LENGTH]

    x_tensor = torch.tensor(sequences)

    # 4. Load the Brain
    print("Waking up the Brain...")
    device = torch.device('cpu') 
    model = WiFiShieldGRU(input_size=X_scaled_array.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval() 

    # 5. Run Inference
    print("Scanning for anomalies...")
    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = torch.sigmoid(logits).squeeze()
        
        # If it's a single item, probabilities might not be an array, so we handle it:
        if probabilities.dim() == 0:
            probabilities = probabilities.unsqueeze(0)
            
        # Threshold at 50%
        predictions = (probabilities > 0.95).int().numpy()

    # 6. Map Predictions Back to Original Data
    total_attacks = predictions.sum()
    print("\n" + "="*50)
    print("📊 INVESTIGATION RESULTS")
    print("="*50)
    print(f"Total Sequences Scanned: {num_sequences:,}")
    print(f"Total Attacks Detected:  {total_attacks:,}")

    # --- THE PADDING TRICK ---
    # The first 19 packets don't get a prediction because the GRU needs 20 to make its first guess.
    # We pad the beginning with 0s so our AI column aligns perfectly with the original CSV rows.
    padding = np.zeros(SEQUENCE_LENGTH - 1, dtype=int)
    full_predictions = np.concatenate([padding, predictions])

    # Attach the AI's verdict directly to your raw data!
    df['AI_Verdict'] = full_predictions
    df['AI_Verdict'] = df['AI_Verdict'].map({0: 'Normal', 1: 'ATTACK'})

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Full forensic report saved to: {OUTPUT_CSV}")
    print("Next Step: Open this CSV, look at the 'AI_Verdict' column, and see exactly when the AI tripped the alarm!")

if __name__ == "__main__":
    run_blind_investigation()