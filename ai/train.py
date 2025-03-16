import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import chess.pgn
import os
from sklearn.model_selection import train_test_split
from functools import lru_cache
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import pickle
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

class ChessMoveDataset(Dataset):
    def __init__(self, X, y_legal, y_correct, image_dir=None):
        # Convert to torch tensors and move to memory-efficient format
        self.X = torch.from_numpy(X).float()
        self.y_legal = torch.from_numpy(y_legal).float()
        self.y_correct = torch.from_numpy(y_correct).float()
        
        # Image handling
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.image_dir:
            # Load and transform image
            board_state = self.X[idx]
            image_path = os.path.join(self.image_dir, f"position_{idx}.png")
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except:
                # If image not found, create a blank tensor
                image = torch.zeros((3, 224, 224))
            
            return board_state, image, self.y_legal[idx], self.y_correct[idx]
        else:
            return self.X[idx], self.y_legal[idx], self.y_correct[idx]

class ChessMoveValidator(nn.Module):
    def __init__(self, use_images=True):
        super(ChessMoveValidator, self).__init__()
        self.use_images = use_images
        
        # Board representation network
        self.board_net = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Image network (using ResNet18 as feature extractor)
        if use_images:
            self.image_net = models.resnet18(pretrained=True)
            # Freeze the pretrained layers
            for param in self.image_net.parameters():
                param.requires_grad = False
            # Modify the final layer to match our needs
            self.image_net.fc = nn.Linear(512, 512)
        
        # Combined network
        if use_images:
            self.fc = nn.Sequential(
                nn.Linear(1024, 512),  # 512 from board + 512 from image
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 2),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(512, 2),
                nn.Sigmoid()
            )
    
    def forward(self, board_state, image=None):
        # Process board representation
        board_features = self.board_net(board_state)
        
        if self.use_images and image is not None:
            # Process image
            image_features = self.image_net(image)
            # Combine features
            combined = torch.cat([board_features, image_features], dim=1)
            return self.fc(combined)
        else:
            return self.fc(board_features)

@lru_cache(maxsize=1000)
def create_board_representation(board_fen):
    """Convert a chess board to a 8x8x12 tensor representation with caching."""
    board = chess.Board(board_fen)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    board_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_idx = piece_types.index(piece.piece_type)
            color_idx = 0 if piece.color else 6
            board_tensor[rank, file, color_idx + piece_idx] = 1
            
    return board_tensor

def create_move_representation(move, board):
    """Convert a chess move to a 64x64 tensor representation."""
    move_tensor = np.zeros((64, 64), dtype=np.float32)
    from_square = move.from_square
    to_square = move.to_square
    move_tensor[from_square, to_square] = 1
    return move_tensor

def save_board_image(board, save_path, size=(224, 224)):
    """Save a chess board as an image."""
    import chess.svg
    import cairosvg
    from io import BytesIO
    
    # Create SVG
    svg_data = chess.svg.board(board=board, size=400)
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), output_width=size[0], output_height=size[1])
    
    # Save to file
    with open(save_path, 'wb') as f:
        f.write(png_data)

def process_game(game, num_positions, image_dir=None):
    """Process a single game to generate training data and images."""
    X = []
    y_legal = []
    y_correct = []
    
    board = game.board()
    for move in game.mainline_moves():
        if len(X) >= num_positions:
            break
            
        board_tensor = create_board_representation(board.fen())
        legal_moves = list(board.legal_moves)
        
        if legal_moves:
            # Add correct move
            correct_move = move
            correct_move_tensor = create_move_representation(correct_move, board)
            X.append(np.concatenate([board_tensor, correct_move_tensor], axis=-1))
            y_legal.append(1)
            y_correct.append(1)
            
            # Save board image if directory is provided
            if image_dir:
                save_board_image(board, os.path.join(image_dir, f"position_{len(X)-1}.png"))
            
            # Add random legal moves
            random_moves = np.random.choice(legal_moves, min(2, len(legal_moves)), replace=False)
            for random_move in random_moves:
                if random_move != correct_move:
                    random_move_tensor = create_move_representation(random_move, board)
                    X.append(np.concatenate([board_tensor, random_move_tensor], axis=-1))
                    y_legal.append(1)
                    y_correct.append(0)
                    
                    # Save board image
                    if image_dir:
                        save_board_image(board, os.path.join(image_dir, f"position_{len(X)-1}.png"))
            
            # Generate illegal moves
            for _ in range(2):
                random_square = np.random.randint(64)
                random_move = chess.Move(random_square, np.random.randint(64))
                if random_move not in legal_moves:
                    random_move_tensor = create_move_representation(random_move, board)
                    X.append(np.concatenate([board_tensor, random_move_tensor], axis=-1))
                    y_legal.append(0)
                    y_correct.append(0)
                    
                    # Save board image
                    if image_dir:
                        save_board_image(board, os.path.join(image_dir, f"position_{len(X)-1}.png"))
        
        board.push(move)
    
    return X, y_legal, y_correct

def generate_training_data(pgn_file, num_positions=10000, num_workers=4, image_dir=None):
    """Generate training data from PGN file with parallel processing."""
    cache_file = f"data/cache_{num_positions}.pkl"
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print("Loading cached training data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Create image directory if needed
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
    
    X = []
    y_legal = []
    y_correct = []
    
    games = []
    with open(pgn_file) as f:
        while len(X) < num_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    
    # Process games in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for game in games:
            if len(X) >= num_positions:
                break
            futures.append(executor.submit(process_game, game, num_positions - len(X), image_dir))
        
        for future in tqdm(futures, desc="Processing games"):
            game_X, game_y_legal, game_y_correct = future.result()
            X.extend(game_X)
            y_legal.extend(game_y_legal)
            y_correct.extend(game_y_correct)
            
            if len(X) >= num_positions:
                break
    
    # Trim to exact number of positions
    X = np.array(X[:num_positions])
    y_legal = np.array(y_legal[:num_positions])
    y_correct = np.array(y_correct[:num_positions])
    
    # Cache the results
    os.makedirs('data', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((X, y_legal, y_correct), f)
    
    return X, y_legal, y_correct

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_legal = 0
    correct_correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        if len(batch) == 4:  # With images
            board_state, image, batch_y_legal, batch_y_correct = batch
        else:  # Without images
            board_state, batch_y_legal, batch_y_correct = batch
            image = None
        
        board_state = board_state.to(device)
        if image is not None:
            image = image.to(device)
        batch_y_legal = batch_y_legal.to(device)
        batch_y_correct = batch_y_correct.to(device)
        
        optimizer.zero_grad()
        outputs = model(board_state, image)
        
        loss = criterion(outputs[:, 0], batch_y_legal) + criterion(outputs[:, 1], batch_y_correct)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predicted_legal = (outputs[:, 0] > 0.5).float()
        predicted_correct = (outputs[:, 1] > 0.5).float()
        
        correct_legal += (predicted_legal == batch_y_legal).sum().item()
        correct_correct += (predicted_correct == batch_y_correct).sum().item()
        total += batch_y_legal.size(0)
    
    return total_loss / len(train_loader), correct_legal / total, correct_correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_legal = 0
    correct_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if len(batch) == 4:  # With images
                board_state, image, batch_y_legal, batch_y_correct = batch
            else:  # Without images
                board_state, batch_y_legal, batch_y_correct = batch
                image = None
            
            board_state = board_state.to(device)
            if image is not None:
                image = image.to(device)
            batch_y_legal = batch_y_legal.to(device)
            batch_y_correct = batch_y_correct.to(device)
            
            outputs = model(board_state, image)
            loss = criterion(outputs[:, 0], batch_y_legal) + criterion(outputs[:, 1], batch_y_correct)
            
            total_loss += loss.item()
            
            predicted_legal = (outputs[:, 0] > 0.5).float()
            predicted_correct = (outputs[:, 1] > 0.5).float()
            
            correct_legal += (predicted_legal == batch_y_legal).sum().item()
            correct_correct += (predicted_correct == batch_y_correct).sum().item()
            total += batch_y_legal.size(0)
    
    return total_loss / len(val_loader), correct_legal / total, correct_correct / total

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate or load training data
    pgn_file = "data/chess_games.pgn"
    if not os.path.exists(pgn_file):
        print(f"Please provide a PGN file at {pgn_file}")
        return
    
    # Create directories for data and images
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/images', exist_ok=True)
    
    print("Generating training data and images...")
    X, y_legal, y_correct = generate_training_data(
        pgn_file, 
        num_positions=10000,
        image_dir='data/images'
    )
    
    # Split data into training and validation sets
    X_train, X_val, y_legal_train, y_legal_val, y_correct_train, y_correct_val = train_test_split(
        X, y_legal, y_correct, test_size=0.2, random_state=42
    )
    
    # Create data loaders with optimized batch size
    train_dataset = ChessMoveDataset(X_train, y_legal_train, y_correct_train, 'data/images')
    val_dataset = ChessMoveDataset(X_val, y_legal_val, y_correct_val, 'data/images')
    
    # Use larger batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)
    
    # Create model, criterion, and optimizer
    model = ChessMoveValidator(use_images=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    print("Training model...")
    
    for epoch in range(num_epochs):
        train_loss, train_acc_legal, train_acc_correct = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc_legal, val_acc_correct = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc (Legal/Correct): {train_acc_legal:.4f}/{train_acc_correct:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc (Legal/Correct): {val_acc_legal:.4f}/{val_acc_correct:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/chess_move_validator.pth')
            print("Saved best model")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
