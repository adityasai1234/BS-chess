import chess
import numpy as np

def get_ai_move(board):
    legal_moves = list(board.legal_moves)
    if legal_moves:
        move = np.random.choice(legal_moves)
        return f"{chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}"
    
    return "resign"
