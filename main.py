"""Main FastAPI application file"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from models import MoveRequest, NewGameRequest
from engine_manager import engine_manager
import routes
from config import SERVER_HOST, SERVER_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await engine_manager.initialize_engine()
    yield
    # Shutdown
    await engine_manager.close_engine()


# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Register routes
app.get("/", response_class=HTMLResponse)(routes.get_chess_game)
app.get("/engine-status")(routes.engine_status)
app.post("/new-game")(routes.new_game)
app.get("/possible-moves/{game_id}/{square}")(routes.get_possible_moves)
app.post("/make-move")(routes.make_move)
app.post("/engine-move/{game_id}")(routes.engine_move)
app.post("/undo-move/{game_id}")(routes.undo_move)
app.get("/game-state/{game_id}")(routes.get_game_state)
app.delete("/game/{game_id}")(routes.delete_game)
app.get("/games")(routes.list_games)
app.get("/analyze/{game_id}")(routes.analyze_position)
app.get("/difficulty-info")(routes.get_difficulty_info)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Chess vs Stockfish server...")
    print("Engine Configuration:")
    print("- Hash: 256MB")
    print("- Threads: Up to 4 CPU cores")
    print("- Skill Levels: 0-20 based on difficulty")
    print("- Time Limits: 0.5s to 6.0s based on difficulty")
    print("- Search Depth: 6 to 20 moves based on difficulty")
    print("\nMake sure Stockfish is installed on your system:")
    print("- Ubuntu/Debian: sudo apt-get install stockfish")
    print("- macOS: brew install stockfish") 
    print("- Windows: Download from https://stockfishchess.org/download/")
    print(f"\nServer will be available at: http://localhost:{SERVER_PORT}")
    print(f"Analysis endpoint: http://localhost:{SERVER_PORT}/analyze/{{game_id}}")
    print(f"Difficulty info: http://localhost:{SERVER_PORT}/difficulty-info")
    
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)