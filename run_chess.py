from ai.mainai import ChessAI
import chess
import time
import numpy as np
import sys

def print_board(board):
    """Print the current board state."""
    print("\n")
    print(board)

def parse_move(move_str):
    """Parse a move string into a chess.Move object."""
    try:
        from_square = chess.parse_square(move_str.split('-')[0])
        to_square = chess.parse_square(move_str.split('-')[1])
        return chess.Move(from_square, to_square)
    except (ValueError, IndexError):
        return None

def get_human_move(board):
    """Get and validate a human move."""
    while True:
        try:
            move_str = input("\nYour move (e.g., e2-e4): ").strip()
            if move_str.lower() == 'quit':
                return None
            
            move = parse_move(move_str)
            if move is None:
                print("Invalid format! Use format: e2-e4")
                continue
            
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Try again.")
        except KeyboardInterrupt:
            print("\nGame ended by user.")
            return None
        except Exception as e:
            print(f"Error processing move: {e}")
            print("Please try again.")

def get_ai_move(board, ai):
    """Get AI move and handle illegal moves."""
    try:
        # Get AI's move
        move_str = ai.get_ai_move(board)
        if not move_str:
            return None
            
        print(f"\nAI's move: {move_str}")
        
        # Parse move
        from_square, to_square = move_str.split('-')
        from_square = chess.parse_square(from_square)
        to_square = chess.parse_square(to_square)
        
        # Create move object
        move = chess.Move(from_square, to_square)
        
        # Check if move is legal
        if move in board.legal_moves:
            print("AI made a legal move! Making it illegal...")
            # Make the move illegal by moving to a random square
            while True:
                random_to = np.random.randint(64)
                if random_to != to_square:
                    move = chess.Move(from_square, random_to)
                    if move not in board.legal_moves:
                        print(f"AI's illegal move: {chess.square_name(from_square)}-{chess.square_name(random_to)}")
                        return move
        
        return move
        
    except Exception as e:
        print(f"Error processing AI move: {e}")
        return None

def main():
    """Main game loop."""
    try:
        # Initialize AI
        print("Initializing AI...")
        ai = ChessAI()
        
        # Initialize board
        board = chess.Board()
        
        print("\nWelcome to Chess AI!")
        print("You are playing as White.")
        print("Enter moves in the format 'e2-e4'")
        print("Type 'quit' to exit")
        
        while not board.is_game_over():
            # Print current board
            print("\nCurrent board position:")
            print(board)
            
            # Get human move
            move = get_human_move(board)
            if move is None:
                print("\nGame ended by user.")
                break
            
            # Make human move
            board.push(move)
            
            # Check if game is over after human move
            if board.is_game_over():
                print("\nGame Over!")
                print(f"Result: {board.result()}")
                break
            
            # Get AI move
            move = get_ai_move(board, ai)
            if move is None:
                print("\nAI failed to make a move. Game ended.")
                break
            
            # Make AI move
            board.push(move)
            
            # Check if game is over after AI move
            if board.is_game_over():
                print("\nGame Over!")
                print(f"Result: {board.result()}")
                break
        
    except KeyboardInterrupt:
        print("\nGame ended by user.")
    except Exception as e:
        print(f"\nGame ended due to error: {e}")
    finally:
        print("\nThanks for playing!")

if __name__ == "__main__":
    main() 