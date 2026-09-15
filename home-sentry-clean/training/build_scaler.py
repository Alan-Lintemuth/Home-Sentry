import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

# --- 1. CONFIGURATION ---
# List all the CSVs you want to merge to build the ultimate scaler
INPUT_CSVS = [
    "cleaned_AWID3_Attack_22.csv", 
    "cleaned_AWID3_Attack_23.csv",
    "cleaned_AWID3_Normal_1.csv",
    "cleaned_AWID3_Normal_2.csv",
    "cleaned_AWID3_Normal_3.csv",
    "home_attack_1.csv", 
    "home_attack_2.csv", 
    "home_normal4_V.csv", 
    "home_normal5_V.csv", 
    "home_normal6_V.csv",
    "home_normal7_V.csv",
    "home_normal8_V.csv"
]
OUTPUT_CSV = "tensor_ready_both_dataset.csv"
SCALER_SAVE_PATH = "models/mega_both_scaler.pkl"

def scale_awid_data():
    print("Loading and combining CSVs...")
    dataframes = []
    
    # 1. Iterate through and load all CSVs
    for file in INPUT_CSVS:
        try:
            print(f" -> Reading {file}...")
            df_part = pd.read_csv(file)
            dataframes.append(df_part)
        except FileNotFoundError:
            print(f" ⚠️ WARNING: {file} not found. Skipping...")
            
    # 2. Stack them vertically into one massive dataset
    df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ All data merged. Total combined rows: {len(df):,}")

    # 3. Separate the Label so it remains human-readable
    X = df.drop(columns=['Label'])
    y = df['Label']

    print("Building and applying complex Scikit-Learn scaling...")

    # --- 2. DEFINE THE SCALING GROUPS ---
    # Group 1: RobustScaler (For data with extreme outliers / bursts)
    robust_features = [
        'packet_frequency', 'frame.len', 'frame.cap_len', 
        'frame.time_delta', 'frame.time_relative', 'wlan.duration',
        'radiotap.length', 'wlan.fixed.timestamp', 'window_time_delta'
    ]

    # Group 2: MinMaxScaler (For categorical IDs to preserve exactly 0.0 - 1.0)
    minmax_features = [
        'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.version', 
        'wlan.frag', 'wlan.seq', 'wlan.qos.tid', 'wlan.qos.priority',
        'wlan.fixed.reason_code', 'wlan.fixed.status_code', 
        'wlan.fixed.auth.alg', 'wlan.fixed.auth_seq', 
        'eapol.type', 'eap.code', 'tls.record.version'
    ]

    # Group 3: StandardScaler (For physical radio waves and Gaussian distributions)
    standard_features = [
        'radiotap.dbm_antsignal', 'radiotap.datarate', 'radiotap.channel.freq'
    ]

    # *Note: All remaining features (the dozens of 1/0 boolean flags) 
    # will automatically be ignored by the scalers because we use remainder='passthrough'.

    # --- 3. BUILD THE TRANSFORMER ---
    print("Building the multi-scaler pipeline...")
    
    # We must only include columns that actually exist in the dataframe to avoid errors
    existing_robust = [col for col in robust_features if col in X.columns]
    existing_minmax = [col for col in minmax_features if col in X.columns]
    existing_standard = [col for col in standard_features if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), existing_robust),
            ('minmax', MinMaxScaler(), existing_minmax),
            ('standard', StandardScaler(), existing_standard)
        ],
        remainder='passthrough'
    )

    # --- 4. FIT, TRANSFORM, AND SAVE ---
    print("Fitting weights and transforming data...")
    X_scaled_array = preprocessor.fit_transform(X)

    # Scikit-learn outputs a raw NumPy array, so we map the column names back
    # The ColumnTransformer rearranges column order, so we pull the new order directly from it
    new_column_order = (
        existing_robust + 
        existing_minmax + 
        existing_standard + 
        [col for col in X.columns if col not in existing_robust + existing_minmax + existing_standard]
    )
    
    final_scaled_df = pd.DataFrame(X_scaled_array, columns=new_column_order)
    
    # Re-attach the human-readable Label
    final_scaled_df['Label'] = y.values

    # Save the Dataset
    final_scaled_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Final Tensor-ready dataset saved to: {OUTPUT_CSV}")

    # Save the Scaler Object for the Live Shield
    joblib.dump(preprocessor, SCALER_SAVE_PATH)
    print(f"Scaler logic successfully saved to: {SCALER_SAVE_PATH}")

if __name__ == "__main__":
    scale_awid_data()