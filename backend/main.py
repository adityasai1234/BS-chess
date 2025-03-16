from typing import Annotated, Any
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/app", StaticFiles(directory="frontend"), name="static")
api = FastAPI(root_path="/api")

@api.post("/move")
async def move(board: Annotated[Any, Body()]):
    position = board["position"]
    return {
        "move": "e7e6"
    }

app.mount("/api", api)
