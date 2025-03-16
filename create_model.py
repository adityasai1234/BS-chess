import torch
import torch.nn as nn

class ChessMoveValidator(nn.Module):
    def __init__(self):
        super(ChessMoveValidator, self).__init__()
        # Input: 8x8x9 (board state) + 8x8x1 (move) = 8x8x10
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 outputs: legality and correctness
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, 10, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def main():
    # Create model instance
    model = ChessMoveValidator()
    
    # Save the model
    torch.save(model.state_dict(), 'models/chess_move_validator.pth')
    print("Model saved successfully to models/chess_move_validator.pth")

if __name__ == "__main__":
    main() 