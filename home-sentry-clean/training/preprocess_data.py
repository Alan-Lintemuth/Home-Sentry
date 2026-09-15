import pandas as pd
import numpy as np
import os
import warnings 

# Mute the Pandas fragmentation warnings for clean console output
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

# --- 1. CONFIGURATION ---
# We hardcode your Home Router's MAC here so the script can establish directionality.
ROUTER_MAC = "0c:9d:92:54:fe:34"  # Replace with your actual router MAC
INPUT_CSV = r"B:\AWID3\CSV\1.Deauth\Deauth_3.csv"   # The raw capture file
OUTPUT_CSV = "cleaned_AWID3_Normal_3.csv"

def preprocess_awid_data(filepath):
    print(f"Loading raw data from {filepath}...")
    # low_memory=False prevents Pandas from guessing datatypes chunk-by-chunk
    df = pd.read_csv(filepath, low_memory=False)

    # Standardize string cases just in case Tshark got weird
    if 'wlan.ra' in df.columns and 'wlan.ta' in df.columns:
        df['wlan.ra'] = df['wlan.ra'].astype(str).str.lower()
        df['wlan.ta'] = df['wlan.ta'].astype(str).str.lower()

    # --- 2. FEATURE ENGINEERING (The 4 New Behaviors) ---
    print("Engineering behavioral features...")
    
    # Behavior 1: Is it a broadcast frame?
    df['is_broadcast'] = (df['wlan.ra'] == "ff:ff:ff:ff:ff:ff").astype(int)

    # Behavior 2: Is it going to the Router?
    df['is_to_ap'] = (df['wlan.ra'] == ROUTER_MAC.lower()).astype(int)

    # Behavior 3: Is it coming from the Router?
    df['is_from_ap'] = (df['wlan.ta'] == ROUTER_MAC.lower()).astype(int)

    # Behavior 4 & 5: Window Delta and Packet Frequency
    print("Calculating window time delta and packet frequency...")
    
    # Ensure precision is maintained for the math
    df['frame.time_delta'] = df['frame.time_delta'].astype(float)
    
    # Sum the time deltas over the last 9 packets per MAC address
    df['window_time_delta'] = df.groupby('wlan.ta')['frame.time_delta'].transform(
        lambda x: x.rolling(window=19, min_periods=1).sum()
    )
    df['window_time_delta'] = df['window_time_delta'].fillna(0.0)

    # Frequency = Number of Packets (10) / Time taken for those packets
    df['packet_frequency'] = np.where(
        df['window_time_delta'] > 0.000001,  # Safe threshold to avoid divide-by-zero
        20.0 / df['window_time_delta'], 
        0.0  # Default to 0 for the very first packet seen
    )
        
# --- 3. ENFORCING THE STATIC SCHEMA (Do this FIRST!) ---
    print("Filtering down to the static 58-column schema...")

    # This list MUST contain exactly 58 input features + 1 Label = 59 items.
    static_features = [
        
        # --- Engineered Behavioral Features (5) ---
        'is_broadcast', 
        'is_to_ap', 
        'is_from_ap', 
        'packet_frequency',
        'window_time_delta',
        
        # --- Frame & Time Metadata (5) ---
        'frame.len', 
        'frame.cap_len', 
        'frame.time_delta',
        'frame.time_relative',
        'wlan.duration',
        
        # --- Radio / Physical Layer (Radiotap) (14) ---
        'radiotap.length',
        'radiotap.present.tsft',
        'radiotap.present.flags',
        'radiotap.present.rate',
        'radiotap.present.channel',
        'radiotap.present.fhss',
        'radiotap.present.dbm_antsignal',
        'radiotap.present.dbm_antnoise',
        'radiotap.present.lock_quality',
        'radiotap.present.tx_attenuation',
        'radiotap.present.db_tx_attenuation',
        'radiotap.dbm_antsignal', 
        'radiotap.datarate', 
        'radiotap.channel.freq',
        
        # --- WLAN Frame Control (MAC Header) (11) ---
        'wlan.fc.type', 
        'wlan.fc.subtype', 
        'wlan.fc.version',
        'wlan.fc.tods',
        'wlan.fc.fromds',
        'wlan.fc.frag',
        'wlan.fc.retry', 
        'wlan.fc.pwrmgt', 
        'wlan.fc.moredata',
        'wlan.fc.protected',
        'wlan.fc.order',
        
        # --- WLAN Sequence & Fragment (3) ---
        'wlan.frag',
        'wlan.seq',
        'wlan.ba.control.ackpolicy',
        
        # --- WLAN Quality of Service (QoS) (7) ---
        'wlan.qos.tid',
        'wlan.qos.priority',
        'wlan.qos.eosp',
        'wlan.qos.ack',
        'wlan.qos.amsdupresent',
        'wlan.qos.buf_state_indicated',
        'wlan.qos.bit4',
        
        # --- WLAN Fixed Parameters & Management (8) ---
        'wlan.fixed.capabilities.ess',
        'wlan.fixed.capabilities.ibss',
        'wlan.fixed.capabilities.privacy',
        'wlan.fixed.capabilities.spec_man',
        'wlan.fixed.capabilities.short_slot_time',
        'wlan.fixed.reason_code',
        'wlan.fixed.status_code',
        'wlan.fixed.timestamp',
        
        # --- Enterprise Authentication & Security (5) ---
        'wlan.fixed.auth.alg',
        'wlan.fixed.auth_seq',
        'eapol.type',
        'eap.code',
        'tls.record.version',
        
        # --- Target Label (1) ---
        # 0 for Normal, 1 for Attack
        'Label' 
    ]

    # Generate missing columns as NaN initially
    for col in static_features:
        if col not in df.columns:
            df[col] = np.nan
            
    # Lock exact order AND drop all the 100+ junk columns
    # The .copy() command instantly fixes the Fragmentation warning!
    df = df[static_features].copy()

# --- 4. FORCE NUMERIC (The Hex Fix) ---
    print("Forcing all feature columns to numeric types...")
    # We must temporarily separate the 'Label' column because if it contains text 
    # (like "normal" or "deauth"), forcing it to numeric will destroy it.
    label_col = df['Label'].copy()
    df = df.drop(columns=['Label'])
    
    # Convert everything else to numbers. If Tshark left weird text (like hex '0x08'), 
    # errors='coerce' turns it into a NaN (blank) so we can fill it with 0 in the next step.
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Put the Label column back
    df['Label'] = label_col

    # --- 5. NULL IMPUTATION ---
    print("Filling missing values with 0...")
    # Now that ONLY numeric columns exist, this will run perfectly without crashing
    df.fillna(0, inplace=True)

    return df

# --- RUN THE PIPELINE ---
if __name__ == "__main__":
    final_df = preprocess_awid_data(INPUT_CSV)
    print(f"\nFinal Dataset Shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns.")
    
    # --- SAFE SAVE LOGIC ---
    base_name, ext = os.path.splitext(OUTPUT_CSV)
    counter = 1
    safe_output = OUTPUT_CSV
    
    # Check if file exists, increment counter until we find a free name
    while os.path.exists(safe_output):
        safe_output = f"{base_name}_{counter}{ext}"
        counter += 1
        
    final_df.to_csv(safe_output, index=False)
    print(f"Data perfectly formatted and saved safely to: {safe_output}")