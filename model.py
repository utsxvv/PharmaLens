import torch
import torch.nn as nn

from charset import NUM_CLASSES


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(


            # Block 1 : [B, 1, 32, 128]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),    
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          
            # [B, 64, 16, 64]

            # Block 2 : [B, 128, 16, 64]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          
            # [B, 128, 8, 32]

            # Block 3 :  [B, 256, 8, 32]
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),               
            # [B, 256, 4, 32]

            # Block 4 [B, 256, 4, 32]
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            # [B, 512, 2, 32]              

            # Block 5 [B, 512, 2, 32]
            nn.Conv2d(512, 512, kernel_size=2, padding=0), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # [B, 512, 1, 31]
        )

    def forward(self, x):
        return self.features(x)   

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            bidirectional= True,        
            batch_first  = False,      
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)          
        output    = self.linear(output)   
        return output


class CRNN(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()

        self.cnn    = CNN()
        self.bilstm = BiLSTM(
            input_size  = 512,          
            hidden_size = hidden_size,
            output_size = NUM_CLASSES,  
        )

    def forward(self, x):
        features = self.cnn(x)              

        batch, channels, height, width = features.size()
        features = features.squeeze(2)      
        features = features.permute(2, 0, 1)  

        output = self.bilstm(features)      

        return output   


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model  = CRNN(hidden_size=256)
    dummy  = torch.randn(2, 1, 32, 128)   
    output = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}   (seq_len, batch, num_classes)")
    print(f"Parameters   : {count_parameters(model):,}")
    print("✅ Model working correctly")
