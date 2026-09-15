import pandas as pd
import joblib

# --- 1. CONFIGURATION ---
INPUT_CSV = "home_normal6_V.csv"  # Put your unseen, unscaled test file here
SCALER_PATH = "models/mega_both_scaler.pkl"      # Point to your already-built scaler
OUTPUT_CSV = "tensor_both_Hnormal_6.csv" # What testSentry.py will read

def apply_saved_scaler():
    print(f"Loading unseen raw data: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Separate the Label so it doesn't get pushed into the math
    X = df.drop(columns=['Label'])
    y = df['Label']

    print(f"Loading trained scaler lens: {SCALER_PATH}...")
    try:
        preprocessor = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Could not find {SCALER_PATH}!")
        return

    print("Translating data (Applying Transform)...")
    # CRITICAL: We use transform() here, NOT fit_transform()
    # This forces the new data to use the exact math from the training phase
    X_scaled_array = preprocessor.transform(X)

    # Reconstruct the column order exactly as the builder script did
    # (Since we are using remainder='passthrough', the scalers shift the columns)
    robust_features = [
        'packet_frequency', 'frame.len', 'frame.cap_len', 
        'frame.time_delta', 'frame.time_relative', 'wlan.duration',
        'radiotap.length', 'wlan.fixed.timestamp', 'window_time_delta'
    ]
    minmax_features = [
        'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.version', 
        'wlan.frag', 'wlan.seq', 'wlan.qos.tid', 'wlan.qos.priority',
        'wlan.fixed.reason_code', 'wlan.fixed.status_code', 
        'wlan.fixed.auth.alg', 'wlan.fixed.auth_seq', 
        'eapol.type', 'eap.code', 'tls.record.version'
    ]
    standard_features = [
        'radiotap.dbm_antsignal', 'radiotap.datarate', 'radiotap.channel.freq'
    ]

    existing_robust = [col for col in robust_features if col in X.columns]
    existing_minmax = [col for col in minmax_features if col in X.columns]
    existing_standard = [col for col in standard_features if col in X.columns]

    new_column_order = (
        existing_robust + 
        existing_minmax + 
        existing_standard + 
        [col for col in X.columns if col not in existing_robust + existing_minmax + existing_standard]
    )

    final_scaled_df = pd.DataFrame(X_scaled_array, columns=new_column_order)
    final_scaled_df['Label'] = y.values

    final_scaled_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Success! Data is now perfectly aligned with the Brain.")
    print(f"Saved to: {OUTPUT_CSV} (Ready for testSentry.py)")

if __name__ == "__main__":
    apply_saved_scaler()