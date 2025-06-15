"""Configuration and constants for the chess application"""

import multiprocessing

# Stockfish configuration
STOCKFISH_HASH_SIZE = 256
STOCKFISH_MAX_THREADS = min(4, max(1, multiprocessing.cpu_count()))
STOCKFISH_PONDER = False
STOCKFISH_CONTEMPT = 0
STOCKFISH_ANALYSE_MODE = False

# Default paths to search for Stockfish
STOCKFISH_PATHS = [
    "/usr/bin/stockfish", "/usr/local/bin/stockfish", 
    "/opt/homebrew/bin/stockfish", "/usr/games/stockfish",
    "C:/Program Files/Stockfish/stockfish.exe",
    "C:/stockfish/stockfish.exe", "./stockfish", "./stockfish.exe"
]

# Difficulty level descriptions
DIFFICULTY_DESCRIPTIONS = {
    1: "Beginner - Very weak play, makes obvious mistakes",
    2: "Novice - Weak play, misses simple tactics",
    3: "Amateur - Basic chess understanding",
    4: "Club Player - Decent tactical awareness",
    5: "Intermediate - Good positional play",
    6: "Advanced Amateur - Strong tactical and positional play",
    7: "Expert - Very strong play, few mistakes",
    8: "Master Level - Excellent play, deep calculation",
    9: "Grandmaster Level - Near-perfect play",
    10: "Super Grandmaster - Maximum strength, deepest analysis"
}

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
TEMPLATE_PATH = "templates/chess.html"