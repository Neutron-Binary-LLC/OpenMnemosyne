import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches)
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, dim)
        )

    def forward(self, x):
        # x shape: (batch, num_patches, dim)
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        return x

class NeuralCritic(nn.Module):
    """
    MLP-Mixer + Lightweight Transformer Encoder
    Targeting ~15M parameters for fast inference (<15ms)
    """
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64, num_blocks=4):
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # MLP-Mixer blocks for feature interaction
        # Assuming sequence length (patches) = 16 for technical/fundamental windows
        self.num_patches = 16
        self.patch_dim = hidden_dim // self.num_patches
        
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(hidden_dim, self.num_patches, hidden_dim // 2, hidden_dim * 2)
            for _ in range(num_blocks // 2)
        ])
        
        # Lightweight Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks // 2)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh() # Normalized corrective signals
        )

    def forward(self, x):
        # x expected: (batch, seq_len, input_dim) -> reduced to (batch, hidden_dim)
        x = self.input_proj(x)
        
        # Interaction via Transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling or take last token
        x = x.mean(dim=1)
        
        # Output 64D corrective vector
        return self.head(x)

def get_neural_critic_model():
    return NeuralCritic()
