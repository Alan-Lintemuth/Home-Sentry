import os
import sys
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import glob
import argparse
import json
from collections import deque
from datetime import datetime  # <-- Added for timestamps

def update_dashboard(status, packets, attacks, confidence, message="Live analysis active.", logs=None):
    if logs is None: logs = []
    with open("dashboard_state.json", "w") as f:
        json.dump({
            "status": status,
            "packets": packets,
            "attacks": attacks,
            "confidence": confidence,
            "message": message,
            "logs": logs
        }, f)
        
from home_sentry.model import WiFiShieldGRU
from home_sentry.features import STATIC_FEATURES, CORE_TSHARK_FIELDS, SEQUENCE_LENGTH

# --- FEATURE SCHEMA ---
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
    'wlan.fixed.auth.alg', 'wlan.fixed.auth_seq', 'eapol.type', 'eap.code', 'tls.record.version'
]

CORE_TSHARK_FIELDS = [f for f in STATIC_FEATURES if f not in [
    'is_broadcast', 'is_to_ap', 'is_from_ap', 'packet_frequency', 'window_time_delta'
]] + ['wlan.ra', 'wlan.ta']

# --- 3. CONFIGURATION & STATE ---
SCALER_PATH = "models/mega_both_scaler.pkl"
MODEL_PATH = "models/sentinel_shield_both_gru.pth" 
TEMP_CSV = "live_raw_temp.csv"
CHUNK_SIZE = "50" 
CONFIDENCE_THRESHOLD = 0.95

def main():
    parser = argparse.ArgumentParser(description="Sentry Live Shield")
    parser.add_argument("--interface", required=True, help="Monitor mode interface")
    parser.add_argument("--target", required=True, help="Target BSSID")
    args = parser.parse_args()

    # We map the argparse arguments directly to your variables here
    interface = args.interface
    router_mac = args.target.lower()
    
    print("="*50)
    print("🛡️  INITIALIZING SENTINEL LIVE SHIELD  🛡️")
    print("="*50)
    print(f"\n[1/3] Target Locked: {router_mac} on {interface}")

    print("[2/3] Waking up the Brain...")
    try:
        scaler = joblib.load(SCALER_PATH)
        device = torch.device('cpu') 
        model = WiFiShieldGRU(input_size=len(STATIC_FEATURES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load Brain or Lens. {e}")
        sys.exit(1)

    print("[3/3] Initializing Memory Buffers...")
    # The AI's math buffer (20 packets)
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) 
    # The Dashcam buffer (20 before, 20 during, 20 after = 60 packets)
    dashcam_buffer = deque(maxlen=60)
    ui_log_buffer = deque(maxlen=100)
    
    total_packets_scanned = 0
    total_attacks_detected = 0
    
    # State variable for the Dashcam
    capture_countdown = 0 

    tshark_cmd = [
        "sudo", "tshark", "-i", interface, "-c", CHUNK_SIZE, 
        "-T", "fields", "-E", "separator=,", "-E", "occurrence=f", "-E", "header=y"
    ]
    for f in CORE_TSHARK_FIELDS: tshark_cmd.extend(["-e", f])

    print("\n✅ SYSTEM ONLINE. Listening to the airwaves... (Press Ctrl+C to stop)")
    print("-" * 60)

    try:
        while True:
            # --- EXTRACT ---
            with open(TEMP_CSV, "w") as f:
                subprocess.run(tshark_cmd, stdout=f, stderr=subprocess.DEVNULL, text=True)
            
            if not os.path.exists(TEMP_CSV) or os.path.getsize(TEMP_CSV) < 50:
                continue 
                
            df = pd.read_csv(TEMP_CSV, low_memory=False)
            if df.empty: continue

            # --- TRANSFORM (Engineering) ---
            for col in ['wlan.ra', 'wlan.ta', 'frame.time_delta']:
                if col not in df.columns: df[col] = "0"

            df['wlan.ra'] = df['wlan.ra'].astype(str).str.lower()
            df['wlan.ta'] = df['wlan.ta'].astype(str).str.lower()
            
            df['is_broadcast'] = (df['wlan.ra'] == "ff:ff:ff:ff:ff:ff").astype(int)
            df['is_to_ap'] = (df['wlan.ra'] == router_mac).astype(int)
            df['is_from_ap'] = (df['wlan.ta'] == router_mac).astype(int)
            
            df['frame.time_delta'] = pd.to_numeric(df['frame.time_delta'], errors='coerce').fillna(0)
            df['window_time_delta'] = df.groupby('wlan.ta')['frame.time_delta'].transform(lambda x: x.rolling(19, 1).sum()).fillna(0)
            df['packet_frequency'] = np.where(df['window_time_delta'] > 0, 20.0 / df['window_time_delta'], 0)

            for col in STATIC_FEATURES:
                if col not in df.columns: df[col] = 0
            
            # Final unscaled feature window
            df_clean = df.replace({True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0})
            final_df = df_clean[STATIC_FEATURES].copy()
            final_df = final_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # --- SCALE ---
            X_scaled = scaler.transform(final_df)

            # --- INFERENCE & MEMORY LOGGING ---
            # We zip the human-readable records and scaled math records together
            raw_records = df.to_dict('records')
            
            for raw_row, scaled_row in zip(raw_records, X_scaled):
                dashcam_buffer.append(raw_row)
                sequence_buffer.append(scaled_row)
                total_packets_scanned += 1
                
                # 1. Format packet info for the UI table
                time_str = datetime.now().strftime("%H:%M:%S")
                mac = str(raw_row.get('wlan.ta', 'Unknown'))
                length = raw_row.get('frame.len', '0')
                info_str = f"Length: {length} bytes"
                
                ui_log_buffer.append({
                    "time": time_str,
                    "mac": mac,
                    "info": info_str,
                    "status": "OKAY"
                })
                
                # 2. AI Inference
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    x_tensor = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(x_tensor)
                        prob = torch.sigmoid(logits).item()

                    if prob > CONFIDENCE_THRESHOLD:
                        total_attacks_detected += 1
                        ui_log_buffer[-1]["status"] = "MALICIOUS 🚨" # Flag it in the UI table
                        
                        now = datetime.now().strftime("%H:%M:%S")
                        print(f"🚨 [{now}] ATTACK SIGNATURE DETECTED! Confidence: {prob:.4f}")
                        
                        update_dashboard("Malicious", total_packets_scanned, total_attacks_detected, prob, "Deauth / Malicious signature isolated!", list(ui_log_buffer))
                        
                        if capture_countdown == 0:
                            capture_countdown = 20 
                    else:
                        update_dashboard("Safe", total_packets_scanned, total_attacks_detected, prob, "Traffic profile normal.", list(ui_log_buffer))

                # 3. Handle the Dashcam Snapshot
                if capture_countdown > 0:
                    capture_countdown -= 1
                    if capture_countdown == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_filename = f"forensic_log_{timestamp}.csv"
                        pd.DataFrame(list(dashcam_buffer)).to_csv(log_filename, index=False)
                        print(f"💾 Snapshot Saved: {log_filename} (60-Packet Context Window)")

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("🛑 SHIELD DEACTIVATED: SUMMARY REPORT")
        print("="*50)
        print(f"Total Packets Sniffed: {total_packets_scanned:,}")
        print(f"Malicious Signatures:  {total_attacks_detected:,}")
        
        if os.path.exists(TEMP_CSV): os.remove(TEMP_CSV)
        
        # --- NEW: POST-MISSION FORENSIC MERGE ---
        print("\n[+] Consolidating forensic dashcam logs...")
        log_files = glob.glob("forensic_log_*.csv")
        
        if log_files:
            try:
                # Read all mini-logs into memory
                df_list = [pd.read_csv(file) for file in log_files]
                # Stitch them together
                master_df = pd.concat(df_list, ignore_index=True)
                
                # If snapshots overlapped during a sustained attack, drop the duplicate packets
                master_df = master_df.drop_duplicates()
                
                # Save the master log
                session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                master_filename = f"Master_Forensic_Log_{session_time}.csv"
                master_df.to_csv(master_filename, index=False)
                
                # Clean up the battlefield (delete the mini-logs)
                for file in log_files:
                    os.remove(file)
                    
                print(f"✅ Master log created: {master_filename} ({len(master_df)} unique packets)")
                print(f"🧹 Swept {len(log_files)} temporary snapshot files.")
                os.system("sudo pkill tshark")
            except Exception as e:
                print(f"❌ Error during log consolidation: {e}")
        else:
            print("No forensic logs generated during this session.")
            
        print("\nSystem shutdown complete.")
        sys.exit(0)
      
if __name__ == "__main__":
    main()

