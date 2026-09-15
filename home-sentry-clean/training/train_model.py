import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# --- 1. THE DATASET CLASS ---
class AWID3SequenceDataset(Dataset):
    def __init__(self, dataframe, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.labels = dataframe['Label'].values
        self.features = dataframe.drop('Label', axis=1).values
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        window_features = self.features[idx : idx + self.sequence_length]
        window_label = self.labels[idx + self.sequence_length - 1]
        
        x_tensor = torch.tensor(window_features, dtype=torch.float32)
        y_tensor = torch.tensor(window_label, dtype=torch.float32).unsqueeze(0) 
        
        return x_tensor, y_tensor

from home_sentry.model import WiFiShieldGRU
from home_sentry.features import SEQUENCE_LENGTH

# --- 3. THE TRAINING LOOP ---
def train_model():
    print("Loading scaled dataset...")
    df = pd.read_csv("tensor_ready_both_dataset.csv")  # Updated to match your actual filename
    
    # Map the text labels to binary math (0.0 for Normal, 1.0 for ANY attack)
    df['Label'] = df['Label'].apply(lambda x: 0.0 if x == 'Normal' else 1.0)
    
    print("Building sequences...")
    dataset = AWID3SequenceDataset(df, sequence_length=SEQUENCE_LENGTH)
    # DataLoader feeds the data to the GPU in batches of 64 sequences
    # shuffle=False keeps the timeline perfectly intact
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Detect GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # --- THE CLASS WEIGHT MATH ---
    num_negatives = (df['Label'] == 0.0).sum()
    num_positives = (df['Label'] == 1.0).sum()
    
    # Calculate ratio (prevent division by zero just in case)
    weight_ratio = num_negatives / num_positives if num_positives > 0 else 1.0
    
    print(f"📊 Distribution -> Normal: {num_negatives} | Attack: {num_positives}")
    print(f"⚖️ Applying Attack Penalty Weight: {weight_ratio:.2f}x")
    
    # Convert weight to a tensor and push to GPU/CPU
    pos_weight_tensor = torch.tensor([weight_ratio], dtype=torch.float32).to(device)

    # Initialize Model
    model = WiFiShieldGRU(input_size=58).to(device)
    # BCEWithLogitsLoss with pos_weight applied
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) 
    # Adam Optimizer adjusts the weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    EPOCHS = 10
    
    print("Starting Training...")
    model.train() # Put model in training mode (enables dropout)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_x, batch_y in dataloader:
            # Move tensors to GPU/CPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 1. Forward Pass: Ask the AI for its prediction
            predictions = model(batch_x)
            # 2. Calculate the Loss (How wrong was the AI?)
            loss = criterion(predictions, batch_y)
            # 3. Backward Pass: Calculate the corrections
            optimizer.zero_grad() # Clear old gradients
            loss.backward()       # Math magic (Backpropagation)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Convert raw logits to probability, then to 0 or 1
            probs = torch.sigmoid(predictions)
            predicted_classes = (probs > 0.5).float()
            correct_predictions += (predicted_classes == batch_y).sum().item()
            total_samples += batch_y.size(0)
            
        epoch_accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {epoch_accuracy:.2f}%")

    # --- 4. SAVE THE WEAPONIZED WEIGHTS ---
    SAVE_PATH = "models/sentinel_shield_both_gru.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Training Complete! Model weights successfully saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_model()