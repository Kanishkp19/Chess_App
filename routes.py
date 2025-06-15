"""API routes for the chess application"""

import chess
import asyncio
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from models import MoveRequest, NewGameRequest
from database import GameDatabase
from engine_manager import engine_manager
from utils import get_game_result, get_engine_limits, get_difficulty_description
from config import TEMPLATE_PATH

# Initialize database
db = GameDatabase()


async def get_chess_game():
    """Serve the chess game HTML page"""
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chess game template not found")


async def engine_status():
    """Get engine status and information"""
    success, result = await engine_manager.test_engine()
    
    if success:
        return {
            "available": True,
            "engine_info": result,
            "test_successful": True,
            "analysis_sample": result.get("analysis_sample", "N/A")
        }
    else:
        return {
            "available": False,
            "engine_info": None,
            "error": result
        }


async def new_game(request: NewGameRequest):
    """Create a new chess game"""
    if not engine_manager.engine:
        raise HTTPException(status_code=503, detail="Stockfish engine not available")
    
    # Validate difficulty
    if not 1 <= request.difficulty <= 10:
        raise HTTPException(status_code=400, detail="Difficulty must be between 1 and 10")
    
    game_id = db.create_game(request.player_color, request.difficulty)
    game = db.get_game(game_id)
    
    return {
        "game_id": game_id,
        "board_fen": game["board"].fen(),
        "player_color": request.player_color,
        "difficulty": request.difficulty,
        "game_over": False,
        "result": None,
        "move_history": []
    }


async def get_possible_moves(game_id: str, square: str):
    """Get possible moves from a square"""
    game = db.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        square_index = chess.parse_square(square)
        moves = [chess.square_name(move.to_square) for move in game["board"].legal_moves 
                if move.from_square == square_index]
        return {"moves": moves}
    except:
        return {"moves": []}


async def make_move(request: MoveRequest):
    """Make a player move"""
    game = db.get_game(request.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        move = chess.Move.from_uci(request.move)
        if move not in game["board"].legal_moves:
            raise HTTPException(status_code=400, detail="Invalid move")
        
        game["board"].push(move)
        game["move_history"].append(request.move)
        
        game_over = game["board"].is_game_over()
        result = None
        
        if game_over:
            result = get_game_result(game["board"], game["player_color"])
        
        return {
            "board_fen": game["board"].fen(),
            "player_color": game["player_color"],
            "difficulty": game["difficulty"],
            "game_over": game_over,
            "result": result,
            "move_history": game["move_history"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making move: {str(e)}")


async def engine_move(game_id: str):
    """Get engine move"""
    if not engine_manager.engine:
        raise HTTPException(status_code=503, detail="Stockfish engine not available")
        
    game = db.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if game["board"].is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")
    
    try:
        move = await engine_manager.get_engine_move(game["board"], game["difficulty"])
        
        # Apply the move
        game["board"].push(move)
        game["move_history"].append(move.uci())
        
        # Check game state
        game_over = game["board"].is_game_over()
        result_text = None
        
        if game_over:
            result_text = get_game_result(game["board"], game["player_color"])
        
        print(f"Engine played: {move.uci()}")
        
        return {
            "board_fen": game["board"].fen(),
            "player_color": game["player_color"],
            "difficulty": game["difficulty"],
            "game_over": game_over,
            "result": result_text,
            "last_move": move.uci(),
            "move_history": game["move_history"]
        }
        
    except asyncio.TimeoutError:
        limits = get_engine_limits(game["difficulty"])
        timeout = limits["time"] + 10.0
        raise HTTPException(status_code=408, detail=f"Stockfish timeout after {timeout}s - try reducing difficulty")
    except Exception as e:
        print(f"Engine error: {e}")
        raise HTTPException(status_code=500, detail=f"Stockfish error: {str(e)}")


async def undo_move(game_id: str):
    """Undo the last move(s)"""
    game = db.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if len(game["move_history"]) == 0:
        raise HTTPException(status_code=400, detail="No moves to undo")
    
    try:
        # Undo player move
        if game["board"].move_stack:
            game["board"].pop()
            game["move_history"].pop()
        
        # Undo engine move if it exists and it's the engine's turn to be undone
        if (len(game["move_history"]) > 0 and 
            len(game["board"].move_stack) > 0 and 
            ((game["player_color"] == "white" and game["board"].turn == False) or 
             (game["player_color"] == "black" and game["board"].turn == True))):
            game["board"].pop()
            game["move_history"].pop()
        
        return {
            "board_fen": game["board"].fen(),
            "player_color": game["player_color"],
            "difficulty": game["difficulty"],
            "game_over": False,
            "result": None,
            "move_history": game["move_history"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error undoing move: {str(e)}")


async def get_game_state(game_id: str):
    """Get current game state"""
    game = db.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    board = game["board"]
    game_over = board.is_game_over()
    result = None
    
    if game_over:
        result = get_game_result(board, game["player_color"])
    
    return {
        "board_fen": board.fen(),
        "player_color": game["player_color"],
        "difficulty": game["difficulty"],
        "game_over": game_over,
        "result": result,
        "move_history": game["move_history"],
        "turn": "white" if board.turn else "black"
    }


async def delete_game(game_id: str):
    """Delete a game"""
    if not db.get_game(game_id):
        raise HTTPException(status_code=404, detail="Game not found")
    
    db.delete_game(game_id)
    return {"message": "Game deleted successfully"}


async def list_games():
    """List all games"""
    return db.get_all_games_info()


async def analyze_position(game_id: str, depth: int = 12):
    """Get Stockfish analysis of current position"""
    if not engine_manager.engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    game = db.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        analysis, used_depth = await engine_manager.analyze_position(game["board"], depth)
        
        return {
            "analysis": analysis,
            "depth": used_depth,
            "position_fen": game["board"].fen()
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


async def get_difficulty_info():
    """Get information about difficulty levels"""
    difficulty_info = {}
    for level in range(1, 11):
        limits = get_engine_limits(level)
        difficulty_info[level] = {
            "skill_level": limits["skill_level"],
            "time_seconds": limits["time"],
            "depth": limits["depth"],
            "description": get_difficulty_description(level)
        }
    
    return {"difficulty_levels": difficulty_info}