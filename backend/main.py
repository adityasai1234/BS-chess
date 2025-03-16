from typing import Annotated, Any
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
import bot 
import chess
from google import genai
import os
import dotenv

dotenv.load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)

app = FastAPI()

app.mount("/app", StaticFiles(directory="frontend"), name="static")
api = FastAPI(root_path="/api")

opponent_side = 'b'

@api.post("/move")
async def move(board: Annotated[Any, Body()]):
    if opponent_side == 'w':
        # Flip board
        board['position'].reverse()
        move = bot.get_ai_move(chess.Board(make_fen(board['position'])).mirror())
    else:
        move = bot.get_ai_move(chess.Board(make_fen(board['position'])))

    return {
        'move': move.replace("-", "")
    }

@api.get('/usuck')
async def usuck():
    # Use gemini api key to generate a discouraging message
    return {
        'message': client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Write a short piece of trashtalk to discourage an opponent in chess middle game. Maximum 2 sentences long.',
        ).text
    } 

@api.put('/side')
async def side():
    global opponent_side 
    opponent_side = 'w' if opponent_side == 'b' else 'b'

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
    return result[:-1] + f" {opponent_side} - - 0 1"

app.mount("/api", api)
