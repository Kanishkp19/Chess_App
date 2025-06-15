"""Utility functions for the chess application"""

import os
import shutil
import subprocess
from config import STOCKFISH_PATHS, DIFFICULTY_DESCRIPTIONS


def find_stockfish():
    """Find Stockfish executable with better detection"""
    stockfish_cmd = shutil.which("stockfish")
    if stockfish_cmd:
        return stockfish_cmd
    
    for path in STOCKFISH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    try:
        if os.name != 'nt':
            result = subprocess.run(['whereis', 'stockfish'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                paths = result.stdout.split()[1:]
                for path in paths:
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        return path
    except:
        pass
    
    return None


def get_engine_limits(difficulty: int):
    """Get appropriate engine limits based on difficulty level"""
    if difficulty <= 2:
        # Beginner: Use very low skill level with limited time
        return {
            "skill_level": max(0, difficulty * 2),  # 0, 2, 4
            "time": 0.5,
            "depth": 6
        }
    elif difficulty <= 4:
        # Novice: Low skill level with moderate time
        return {
            "skill_level": 3 + difficulty,  # 6, 7, 8
            "time": 1.0,
            "depth": 8
        }
    elif difficulty <= 6:
        # Intermediate: Moderate skill with good time
        return {
            "skill_level": 8 + (difficulty - 4) * 2,  # 10, 12, 14
            "time": 2.0,
            "depth": 10 + (difficulty - 4)  # 10, 11, 12
        }
    elif difficulty <= 8:
        # Advanced: High skill with substantial time
        return {
            "skill_level": 16 + (difficulty - 6),  # 18, 19, 20
            "time": 3.0 + (difficulty - 6) * 0.5,  # 3.0, 3.5, 4.0
            "depth": 14 + (difficulty - 6) * 2  # 14, 16, 18
        }
    else:
        # Expert: Maximum skill with deep analysis
        return {
            "skill_level": 20,  # Maximum skill
            "time": 5.0 + (difficulty - 8) * 1.0,  # 5.0, 6.0 seconds
            "depth": 18 + (difficulty - 8) * 2  # 18, 20 depth
        }


def get_difficulty_description(difficulty: int) -> str:
    """Get human-readable description of difficulty level"""
    return DIFFICULTY_DESCRIPTIONS.get(difficulty, "Unknown difficulty level")


def get_game_result(board, player_color):
    """Get game result text based on board state"""
    if board.is_checkmate():
        return "Checkmate! " + ("You win!" if board.turn != (player_color == "white") else "Stockfish wins!")
    elif board.is_stalemate():
        return "Stalemate! It's a draw."
    elif board.is_insufficient_material():
        return "Draw by insufficient material."
    elif board.is_seventyfive_moves():
        return "Draw by 75-move rule."
    elif board.is_fivefold_repetition():
        return "Draw by repetition."
    else:
        return "Draw."