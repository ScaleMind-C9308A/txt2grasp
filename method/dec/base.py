from torch import nn

class Basev0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.subnet = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 5),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.subnet(x)