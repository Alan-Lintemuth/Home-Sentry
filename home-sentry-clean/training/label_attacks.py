import pandas as pd

# --- 1. CONFIGURATION ---
input_csv = r"B:\AWID3\CSV\1.Deauth\home_attack1.csv"  # Your file from the collector
output_csv = r"B:\AWID3\CSV\1.Deauth\home_training_ready1.csv"

# Define your attack windows in seconds (Relative Time)
# Format: (start_second, end_second)
ATTACK_WINDOWS = [
    (128, 135.1),  # Directed Attack 1
    (290, 297.5),  # Directed Attack 2
    (406, 414),  # Broadcast Attack 1
]

def apply_precision_labels(df):
    print(f"[*] Processing {len(df)} packets...")

    # Ensure we are working with numbers
    df['frame.time_relative'] = pd.to_numeric(df['frame.time_relative'], errors='coerce').fillna(0)
    df['wlan.fc.type'] = pd.to_numeric(df['wlan.fc.type'], errors='coerce').fillna(-1)
    df['wlan.fc.subtype'] = pd.to_numeric(df['wlan.fc.subtype'], errors='coerce').fillna(-1)

    # --- THE FIX: Use Capital 'Label' to overwrite the existing column ---
    df['Label'] = 'Normal'

    # --- THE FILTER LOGIC ---
    # Condition: Management Frame (0) AND Deauth Subtype (12)
    is_deauth = (df['wlan.fc.type'] == 0) & (df['wlan.fc.subtype'] == 12)

    for start, end in ATTACK_WINDOWS:
        # Condition: Time is within the specific attack burst window
        is_in_window = (df['frame.time_relative'] >= start) & (df['frame.time_relative'] <= end)
        
        # Apply 'Attack' label ONLY to deauths inside the window
        df.loc[is_deauth & is_in_window, 'Label'] = 'Attack'
        
        count = len(df[(is_deauth & is_in_window)])
        print(f"  [+] Labeled {count} packets as 'Attack' between {start}s and {end}s")

    # --- LOGGING THE NEGATIVES ---
    # Find deauths that happened OUTSIDE windows (your manual laptop drops)
    normal_deauths = df[(is_deauth) & (df['Label'] == 'Normal')]
    print(f"  [i] Left {len(normal_deauths)} Deauth packets as 'Normal' (Manual drops/Noise)")

    # (Optional safeguard) If a lowercase 'label' somehow snuck in from earlier, drop it:
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    return df

# --- 2. EXECUTION ---
try:
    data = pd.read_csv(input_csv)
    labeled_data = apply_precision_labels(data)
    
    # Save the final dataset
    labeled_data.to_csv(output_csv, index=False)
    print(f"\n✅ Done! Training dataset saved to: {output_csv}")

except FileNotFoundError:
    print(f"[-] Error: Could not find {input_csv}")
except Exception as e:
    print(f"[-] An error occurred: {e}")