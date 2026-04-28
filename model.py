"""
model.py — CRNN (Convolutional Recurrent Neural Network) built from scratch.

Architecture:
    Input Image [1, 32, 128]
         ↓
    CNN  — extracts visual features from the image
         ↓
    BiLSTM — reads the feature sequence left-to-right AND right-to-left
         ↓
    Linear — maps to character probabilities at each time step
         ↓
    CTC Loss — trains without needing character-by-character alignment

Why CRNN?
    - CNN sees local patterns  (curves, lines, dots)
    - LSTM sees sequences      (how characters connect over time)
    - CTC handles alignment    (no need to know where each character starts)

This is exactly the architecture inside EasyOCR, PaddleOCR, and MMOCR.
"""

import torch
import torch.nn as nn

from charset import NUM_CLASSES


class CNN(nn.Module):
    """
    Convolutional backbone — extracts spatial features from the image.

    Think of it as teaching the model to recognize:
        - Curves   → could be 'c', 'o', 'e'
        - Verticals → could be 'l', 'i', '1'
        - Crossings → could be 't', 'f'

    Input:  [batch, 1, 32, 128]   (grayscale image)
    Output: [batch, 512, 1, 32]   (feature map — height collapsed to 1)

    Conv layers explained:
        Conv2d(in_channels, out_channels, kernel_size, padding)
        BatchNorm → stabilizes training
        ReLU      → adds non-linearity (model can learn complex patterns)
        MaxPool   → reduces size, keeps strongest features
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1 — low-level features (edges, corners)
            # Input: [B, 1, 32, 128]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),    # [B, 64, 32, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # [B, 64, 16, 64]

            # Block 2 — mid-level features (strokes, curves)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 16, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # [B, 128, 8, 32]

            # Block 3 — higher-level features (letter parts)
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # [B, 256, 8, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # [B, 256, 8, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),               # [B, 256, 4, 32]

            # Block 4 — high-level features (full character shapes)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # [B, 512, 4, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), # [B, 512, 4, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),               # [B, 512, 2, 32]

            # Block 5 — collapse height to 1
            nn.Conv2d(512, 512, kernel_size=2, padding=0), # [B, 512, 1, 31]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)   # [B, 512, 1, W]


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM — reads the feature sequence in both directions.

    Why bidirectional?
        Reading "Amoxicillin" left→right gives context from previous chars.
        Reading right→left gives context from future chars.
        Combining both = better understanding of each character.

    Input:  [sequence_length, batch, input_size]
    Output: [sequence_length, batch, hidden_size * 2]
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            bidirectional= True,        # reads both directions
            batch_first  = False,       # sequence first for CTC
        )

        # Maps LSTM output to character probabilities
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)          # [seq, batch, hidden*2]
        output    = self.linear(output)   # [seq, batch, num_classes]
        return output


class CRNN(nn.Module):
    """
    Full CRNN model combining CNN + BiLSTM.

    Flow:
        1. CNN extracts features from image  → [B, 512, 1, W]
        2. Reshape for LSTM                  → [W, B, 512]
        3. BiLSTM reads sequence             → [W, B, num_classes]
        4. CTC decodes characters            → "Amoxicillin"

    Args:
        hidden_size : number of LSTM units (default 256)
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()

        self.cnn    = CNN()
        self.bilstm = BiLSTM(
            input_size  = 512,          # matches CNN output channels
            hidden_size = hidden_size,
            output_size = NUM_CLASSES,  # one score per character + blank
        )

    def forward(self, x):
        # Step 1 — CNN feature extraction
        # x: [batch, 1, 32, 128]
        features = self.cnn(x)              # [batch, 512, 1, W]

        # Step 2 — Reshape for LSTM
        # Remove height dimension (it's 1), then rearrange to [W, batch, 512]
        batch, channels, height, width = features.size()
        features = features.squeeze(2)      # [batch, 512, W]
        features = features.permute(2, 0, 1)  # [W, batch, 512]

        # Step 3 — BiLSTM sequence modeling
        output = self.bilstm(features)      # [W, batch, num_classes]

        return output   # raw logits — CTC loss handles the rest


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test to verify model works
    model  = CRNN(hidden_size=256)
    dummy  = torch.randn(2, 1, 32, 128)   # batch of 2 images
    output = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}   (seq_len, batch, num_classes)")
    print(f"Parameters   : {count_parameters(model):,}")
    print("✅ Model working correctly")
