import torch
import chess
import numpy as np
from ai.train import ChessMoveValidator, create_board_representation, create_move_representation

class ChessAI:
    def __init__(self, model_path='models/chess_move_validator.pth', use_images=True):
        """
        Initialize the Chess AI with a trained model.
        
        Args:
            model_path (str): Path to the trained model weights
            use_images (bool): Whether to use image features (should match training)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessMoveValidator(use_images=use_images).to(self.device)
        
        # Load the trained model
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # Set to evaluation mode
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model")
    
    def get_ai_move(self, board, temperature=0.5, top_k=5):
        """
        Get the AI's move for the current board position.
        Makes random moves without checking legality.
        
        Args:
            board (chess.Board): Current chess board position
            temperature (float): Controls randomness in move selection (0.0 to 1.0)
            top_k (int): Number of top moves to consider
            
        Returns:
            str: Move in format "e2-e4"
        """
        try:
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            
            # Create board representation once
            board_tensor = create_board_representation(board.fen())
            board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)
            
            # Evaluate all legal moves
            move_scores = []
            for move in legal_moves:
                try:
                    # Create move representation
                    move_tensor = create_move_representation(move, board)
                    # Reshape move tensor to match board tensor dimensions
                    move_tensor = move_tensor.reshape(8, 8, 1)  # Reshape to 8x8x1
                    move_tensor = torch.from_numpy(move_tensor).float().unsqueeze(0).to(self.device)
                    
                    # Combine board and move tensors
                    input_tensor = torch.cat([board_tensor, move_tensor], dim=-1)
                    
                    # Get model prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        # Use both legality and correctness scores
                        score = (output[0, 0] + output[0, 1]) / 2
                        move_scores.append((move, score.item()))
                except Exception as e:
                    print(f"Error evaluating move {move}: {e}")
                    continue
            
            if not move_scores:
                # If no moves were successfully evaluated, make a random legal move
                move = np.random.choice(legal_moves)
                return f"{chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}"
            
            # Sort moves by score
            move_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply temperature and top-k filtering
            scores = np.array([score for _, score in move_scores[:top_k]])
            scores = np.exp(scores / temperature)
            scores = scores / np.sum(scores)
            
            # Select move based on temperature-scaled probabilities
            selected_idx = np.random.choice(len(scores), p=scores)
            selected_move = move_scores[selected_idx][0]
            
            return f"{chess.square_name(selected_move.from_square)}-{chess.square_name(selected_move.to_square)}"
            
        except Exception as e:
            print(f"Error in get_ai_move: {e}")
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                return f"{chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}"
            return None
    
    def evaluate_position(self, board):
        """
        Evaluate the current board position.
        
        Args:
            board (chess.Board): Current chess board position
            
        Returns:
            float: Position evaluation score (-1 to 1)
        """
        try:
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return 0.0
            
            # Create board representation once
            board_tensor = create_board_representation(board.fen())
            board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)
            
            # Evaluate all legal moves
            move_scores = []
            for move in legal_moves:
                try:
                    # Create move representation
                    move_tensor = create_move_representation(move, board)
                    # Reshape move tensor to match board tensor dimensions
                    move_tensor = move_tensor.reshape(8, 8, 1)  # Reshape to 8x8x1
                    move_tensor = torch.from_numpy(move_tensor).float().unsqueeze(0).to(self.device)
                    
                    # Combine board and move tensors
                    input_tensor = torch.cat([board_tensor, move_tensor], dim=-1)
                    
                    # Get model prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        # Use both legality and correctness scores
                        score = (output[0, 0] + output[0, 1]) / 2
                        move_scores.append(score.item())
                except Exception as e:
                    print(f"Error evaluating move {move}: {e}")
                    continue
            
            if not move_scores:
                return 0.0
            
            return np.mean(move_scores)
            
        except Exception as e:
            print(f"Error in evaluate_position: {e}")
            return 0.0
    
    def get_top_moves(self, board, num_moves=5):
        """
        Get the top N moves for the current position.
        
        Args:
            board (chess.Board): Current chess board position
            num_moves (int): Number of top moves to return
            
        Returns:
            list: List of tuples (move_string, score)
        """
        try:
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return []
            
            # Create board representation once
            board_tensor = create_board_representation(board.fen())
            board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)
            
            # Evaluate all legal moves
            move_scores = []
            for move in legal_moves:
                try:
                    # Create move representation
                    move_tensor = create_move_representation(move, board)
                    # Reshape move tensor to match board tensor dimensions
                    move_tensor = move_tensor.reshape(8, 8, 1)  # Reshape to 8x8x1
                    move_tensor = torch.from_numpy(move_tensor).float().unsqueeze(0).to(self.device)
                    
                    # Combine board and move tensors
                    input_tensor = torch.cat([board_tensor, move_tensor], dim=-1)
                    
                    # Get model prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        # Use both legality and correctness scores
                        score = (output[0, 0] + output[0, 1]) / 2
                        move_scores.append((move, score.item()))
                except Exception as e:
                    print(f"Error evaluating move {move}: {e}")
                    continue
            
            # Sort moves by score and get top N
            move_scores.sort(key=lambda x: x[1], reverse=True)
            top_moves = []
            
            for move, score in move_scores[:num_moves]:
                move_str = f"{chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}"
                top_moves.append((move_str, score))
            
            return top_moves
            
        except Exception as e:
            print(f"Error in get_top_moves: {e}")
            return []
