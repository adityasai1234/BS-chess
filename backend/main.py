from typing import Annotated, Any
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
import ai
import chess

app = FastAPI()

app.mount("/app", StaticFiles(directory="frontend"), name="static")
api = FastAPI(root_path="/api")

model = ai.ChessAI()

@api.post("/move")
async def move(board: Annotated[Any, Body()]):
    return {
        "move": model.get_ai_move(chess.Board(make_fen(board['position']))).replace('-', '')
    }

def make_fen(position: list[list[str]]) -> str:
    result = ""
    for row in position:
        empty = 0
        for cell in row:
            if cell == " ":
                empty += 1
            else:
                if empty:
                    result += str(empty)
                    empty = 0
                result += cell
        if empty:
            result += str(empty)
        result += "/"
    return result[:-1] + " b - - 0 1"

app.mount("/api", api)
