import torch
import torch.nn as nn

class BaseSignalGenerator(nn.Module):
    """
    Skeleton for Temporal Fusion Transformer (TFT) or LSTM-Transformer Hybrid
    """
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=3): # Buy, Sell, Hold
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=4, batch_first=True),
            num_layers=2
        )
        self.head = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        trans_out = self.transformer(lstm_out)
        logits = self.head(trans_out[:, -1, :])
        return torch.softmax(logits, dim=-1)
