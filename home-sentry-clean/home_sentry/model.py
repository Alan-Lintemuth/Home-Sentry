"""PyTorch model architecture shared by training and inference."""
import torch.nn as nn

class WiFiShieldGRU(nn.Module):
    def __init__(self, input_size=58, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        x = self.fc1(out[:, -1, :])
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
