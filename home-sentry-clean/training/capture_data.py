#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings('ignore')

print("🛡️  AWID3 DOMAIN EXPERIMENT: ROBUST COLLECTOR 🛡️")
print("="*50)

# --- 1. CONFIG ---
interface = "wlan0mon"
ROUTER_MAC = input("🎯 Target Router MAC: ").strip().lower()
target_channel = input("📻 Channel: ").strip()
packet_count = input("🔢 Count (Default 5000): ").strip() or "5000"

print("\n🏷️  Label:")
print("  [1] Normal | [2] Attack")
choice = input("Selection: ").strip()
label_input = "Attack" if choice == "2" else "Normal"
output_file = f"home_{label_input.lower()}.csv"

# --- 2. THE SCHEMA ---
STATIC_FEATURES = [
    'is_broadcast', 'is_to_ap', 'is_from_ap', 'packet_frequency', 'window_time_delta',
    'frame.len', 'frame.cap_len', 'frame.time_delta', 'frame.time_relative', 'wlan.duration',
    'radiotap.length', 'radiotap.present.tsft', 'radiotap.present.flags', 'radiotap.present.rate',
    'radiotap.present.channel', 'radiotap.present.fhss', 'radiotap.present.dbm_antsignal',
    'radiotap.present.dbm_antnoise', 'radiotap.present.lock_quality', 'radiotap.present.tx_attenuation',
    'radiotap.present.db_tx_attenuation', 'radiotap.dbm_antsignal', 'radiotap.datarate', 'radiotap.channel.freq',
    'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.version', 'wlan.fc.tods', 'wlan.fc.fromds',
    'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected', 'wlan.fc.order',
    'wlan.frag', 'wlan.seq', 'wlan.ba.control.ackpolicy',
    'wlan.qos.tid', 'wlan.qos.priority', 'wlan.qos.eosp', 'wlan.qos.ack', 'wlan.qos.amsdupresent',
    'wlan.qos.buf_state_indicated', 'wlan.qos.bit4',
    'wlan.fixed.capabilities.ess', 'wlan.fixed.capabilities.ibss', 'wlan.fixed.capabilities.privacy',
    'wlan.fixed.capabilities.spec_man', 'wlan.fixed.capabilities.short_slot_time', 'wlan.fixed.reason_code',
    'wlan.fixed.status_code', 'wlan.fixed.timestamp',
    'wlan.fixed.auth.alg', 'wlan.fixed.auth_seq', 'eapol.type', 'eap.code', 'tls.record.version',
    'Label'
]

# We need these extra to calculate the engineering features
CORE_TSHARK_FIELDS = [f for f in STATIC_FEATURES if f not in [
    'is_broadcast', 'is_to_ap', 'is_from_ap', 'packet_frequency', 'window_time_delta', 'Label'
]] + ['wlan.ra', 'wlan.ta']

# --- 3. THE CAPTURE ---
print(f"\n📡 Locking {interface} to Channel {target_channel}...")
os.system(f"sudo iwconfig {interface} channel {target_channel}")

raw_temp = "raw_temp.csv"
tshark_cmd = ["tshark", "-i", interface, "-c", packet_count, "-T", "fields", "-E", "separator=,", "-E", "occurrence=f", "-E", "header=y"]
for f in CORE_TSHARK_FIELDS: tshark_cmd.extend(["-e", f])

try:
    print(f"📡 Sniffing {packet_count} packets... (Check your interface lights!)")
    with open(raw_temp, "w") as f:
        subprocess.run(tshark_cmd, stdout=f, text=True)
    
    # Check if the file is basically empty
    if not os.path.exists(raw_temp) or os.path.getsize(raw_temp) < 100:
        print("❌ FATAL: TShark didn't capture enough data. Try a different channel or check your interface.")
        sys.exit()

    print("[*] Capture complete. Loading data...")
    df = pd.read_csv(raw_temp, low_memory=False)

    if df.empty:
        print("❌ FATAL: DataFrame is empty. No packets found.")
        sys.exit()

    # --- 4. SAFE ENGINEERING ---
    print("[*] Engineering features...")
    
    # Initialize missing core columns as empty strings so lower() doesn't crash
    for col in ['wlan.ra', 'wlan.ta', 'frame.time_delta']:
        if col not in df.columns: df[col] = "0"

    df['wlan.ra'] = df['wlan.ra'].astype(str).str.lower()
    df['wlan.ta'] = df['wlan.ta'].astype(str).str.lower()
    
    df['is_broadcast'] = (df['wlan.ra'] == "ff:ff:ff:ff:ff:ff").astype(int)
    df['is_to_ap'] = (df['wlan.ra'] == ROUTER_MAC).astype(int)
    df['is_from_ap'] = (df['wlan.ta'] == ROUTER_MAC).astype(int)
    
    df['frame.time_delta'] = pd.to_numeric(df['frame.time_delta'], errors='coerce').fillna(0)
    
    # Calculate packet frequency using rolling window
    df['window_time_delta'] = df.groupby('wlan.ta')['frame.time_delta'].transform(lambda x: x.rolling(19, 1).sum()).fillna(0)
    df['packet_frequency'] = np.where(df['window_time_delta'] > 0, 20.0 / df['window_time_delta'], 0)
    
    df['Label'] = label_input

    # --- 5. SCHEMA CONSOLIDATION ---
    print("[*] Formatting for scaler...")
    # Add any columns that TShark missed (like EAPOL or TLS if no handshake occurred)
    for col in STATIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Ensure everything is a number except Label
    df = df.replace({True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0})
    
    final_df = df[STATIC_FEATURES].copy()
    feat_cols = [c for c in final_df.columns if c != 'Label']
    final_df[feat_cols] = final_df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    final_df.to_csv(output_file, index=False)
    os.remove(raw_temp)
    print(f"✅ SUCCESS: {output_file} created with {len(final_df)} rows.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    if os.path.exists(raw_temp): os.remove(raw_temp)
