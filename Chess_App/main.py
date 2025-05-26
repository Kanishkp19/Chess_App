from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import os
import asyncio
import chess
import chess.engine
import chess.pgn
from dotenv import load_dotenv
import json
from datetime import datetime
import uuid
import sqlite3
import threading
from contextlib import asynccontextmanager
load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Chess AI Backend"}
# Pydantic Models
class MoveAnalysisRequest(BaseModel):
    fen: str
    move: str
    depth: int = 15
    multiPV: int = 3

class GameMoveRequest(BaseModel):
    fen: str
    difficulty: str = "intermediate"
    time_limit: float = 1.0
    opening_book: bool = True

class TutorAnalysisRequest(BaseModel):
    fen: str
    player_move: str
    player_level: str = "intermediate"
    explain_alternatives: bool = True

class GameSessionRequest(BaseModel):
    session_id: str
    player_color: str
    difficulty: str = "intermediate"
    tutoring_enabled: bool = True

class GameMoveSubmission(BaseModel):
    session_id: str
    move: str
    fen_before: str

class UserRegistration(BaseModel):
    username: str
    email: Optional[str] = None

class PuzzleRequest(BaseModel):
    theme: str
    count: int = 10
    min_rating: int = 1200
    max_rating: int = 1800

class PuzzleSolution(BaseModel):
    puzzle_id: str
    moves: List[str]
    time_taken: int = 0

class PositionAnalysisRequest(BaseModel):
    fen: str
    depth: int = 15

class CoachingRequest(BaseModel):
    fen: str
    player_level: str = "intermediate"
    focus_areas: List[str] = []

class SkillAssessmentRequest(BaseModel):
    games: List[Dict]
    time_controls: List[str] = []

class OpeningAnalysisRequest(BaseModel):
    moves: List[str]
    color: str = "white"

# Game session storage
active_games: Dict[str, Dict] = {}

# User Management
class UserManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect('chess_app.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                rating INTEGER DEFAULT 1200,
                games_played INTEGER DEFAULT 0,
                games_won INTEGER DEFAULT 0,
                puzzle_rating INTEGER DEFAULT 1200,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                white_player TEXT NOT NULL,
                black_player TEXT,
                result TEXT,
                moves TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzles (
                id TEXT PRIMARY KEY,
                theme TEXT NOT NULL,
                fen TEXT NOT NULL,
                moves TEXT NOT NULL,
                rating INTEGER DEFAULT 1200,
                solution_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzle_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                correct BOOLEAN DEFAULT FALSE,
                time_taken INTEGER DEFAULT 0,
                moves_made TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: Optional[str] = None):
        user_id = str(uuid.uuid4())
        conn = sqlite3.connect('chess_app.db', check_same_thread=False)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (id, username, email) VALUES (?, ?, ?)
            ''', (user_id, username, email))
            conn.commit()
            return {"user_id": user_id, "username": username, "success": True}
        except sqlite3.IntegrityError:
            return {"success": False, "error": "Username already exists"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
    
    def get_user_stats(self, user_id: str):
        conn = sqlite3.connect('chess_app.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, rating, games_played, games_won, puzzle_rating 
            FROM users WHERE id = ?
        ''', (user_id,))
        
        user_row = cursor.fetchone()
        if not user_row:
            conn.close()
            return {"error": "User not found"}
        
        cursor.execute('''
            SELECT COUNT(*) as total, SUM(CASE WHEN correct THEN 1 ELSE 0 END) as solved
            FROM puzzle_attempts WHERE user_id = ?
        ''', (user_id,))
        
        puzzle_row = cursor.fetchone()
        total_puzzles = puzzle_row[0] if puzzle_row else 0
        solved_puzzles = puzzle_row[1] if puzzle_row else 0
        
        conn.close()
        
        return {
            "username": user_row[0],
            "rating": user_row[1],
            "games_played": user_row[2],
            "games_won": user_row[3],
            "puzzle_rating": user_row[4],
            "puzzles_attempted": total_puzzles,
            "puzzles_solved": solved_puzzles,
            "puzzle_success_rate": (solved_puzzles / total_puzzles * 100) if total_puzzles > 0 else 0
        }

# Enhanced Puzzle Generator
class EnhancedPuzzleGenerator:
    def __init__(self):
        self.sample_puzzles = {
            'tactics': [
                {
                    'fen': 'r1bq1rk1/ppp2ppp/2n5/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQ - 0 6',
                    'moves': ['Bxf7+', 'Kh8', 'Bd5'],
                    'solution_text': 'Fork the king and rook with the bishop sacrifice'
                },
                {
                    'fen': 'r2qkbnr/ppp2ppp/2n5/3pp3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5',
                    'moves': ['Ng5'],
                    'solution_text': 'Attack the f7 square weakness'
                },
                {
                    'fen': '2r3k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1',
                    'moves': ['Re8+', 'Rxe8'],
                    'solution_text': 'Back rank mate pattern'
                }
            ],
            'endgame': [
                {
                    'fen': '8/8/8/8/8/8/k1K5/8 w - - 0 1',
                    'moves': ['Kb3'],
                    'solution_text': 'King and pawn endgame - advance the king to support pawn promotion'
                },
                {
                    'fen': '8/8/8/8/3k4/8/3K4/8 w - - 0 1',
                    'moves': ['Kc3'],
                    'solution_text': 'Opposition in king and pawn endgame'
                }
            ],
            'opening': [
                {
                    'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
                    'moves': ['e5'],
                    'solution_text': 'Control the center with pawns - classical opening principle'
                },
                {
                    'fen': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',
                    'moves': ['Nf3'],
                    'solution_text': 'Develop knights before bishops'
                }
            ]
        }
    
    def generate_puzzles(self, theme: str, count: int = 10, min_rating: int = 1200, max_rating: int = 1800):
        puzzles = []
        theme_puzzles = self.sample_puzzles.get(theme, self.sample_puzzles['tactics'])
        
        for i in range(count):
            puzzle_data = theme_puzzles[i % len(theme_puzzles)]
            puzzle_id = str(uuid.uuid4())
            
            puzzle = {
                'id': puzzle_id,
                'theme': theme,
                'fen': puzzle_data['fen'],
                'moves': puzzle_data['moves'],
                'rating': min_rating + (i * (max_rating - min_rating) // count),
                'solution_text': puzzle_data['solution_text']
            }
            
            puzzles.append(puzzle)
            self.store_puzzle(puzzle)
        
        return puzzles
    
    def store_puzzle(self, puzzle):
        conn = sqlite3.connect('chess_app.db', check_same_thread=False)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO puzzles (id, theme, fen, moves, rating, solution_text)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                puzzle['id'], puzzle['theme'], puzzle['fen'], 
                json.dumps(puzzle['moves']), puzzle['rating'], puzzle['solution_text']
            ))
            conn.commit()
        except Exception as e:
            print(f"Error storing puzzle: {e}")
        finally:
            conn.close()
    
    def validate_solution(self, puzzle_id: str, user_moves: List[str]):
        conn = sqlite3.connect('chess_app.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('SELECT moves, solution_text FROM puzzles WHERE id = ?', (puzzle_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {'valid': False, 'error': 'Puzzle not found'}
        
        solution_moves = json.loads(row[0])
        solution_text = row[1]
        
        if len(user_moves) >= len(solution_moves):
            if user_moves[:len(solution_moves)] == solution_moves:
                return {
                    'valid': True,
                    'complete': True,
                    'solution_text': solution_text
                }
        
        return {
            'valid': False,
            'complete': False,
            'hint': "Check the tactical pattern again"
        }
    
    def record_attempt(self, puzzle_id: str, user_id: str, correct: bool, time_taken: int, moves: List[str]):
        conn = sqlite3.connect('chess_app.db', check_same_thread=False)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO puzzle_attempts (puzzle_id, user_id, correct, time_taken, moves_made)
                VALUES (?, ?, ?, ?, ?)
            ''', (puzzle_id, user_id, correct, time_taken, json.dumps(moves)))
            conn.commit()
        except Exception as e:
            print(f"Error recording attempt: {e}")
        finally:
            conn.close()

# Stockfish Engine Manager
class StockfishManager:
    def __init__(self):
        self.engine = None
        self.engine_path = os.getenv("STOCKFISH_PATH", "stockfish")
    
    async def initialize(self):
        try:
            if not self.engine:
                # Fixed: Use the correct async method
                transport, self.engine = await chess.engine.popen_uci(self.engine_path)
                print("Stockfish engine initialized successfully!")
        except FileNotFoundError:
            print(f"Stockfish not found at path: {self.engine_path}")
            print("Please install Stockfish or set the correct path in STOCKFISH_PATH environment variable")
            # For development, we'll create a mock engine instead of failing
            self.engine = MockStockfishEngine()
            print("Using mock engine for development")
        except Exception as e:
            print(f"Failed to initialize Stockfish: {e}")
            print("Using mock engine for development")
            self.engine = MockStockfishEngine()
    
    async def analyze_move(self, board: chess.Board, move: chess.Move, depth: int = 15, multiPV: int = 3):
        if isinstance(self.engine, MockStockfishEngine):
            return await self.engine.analyze_move(board, move, depth, multiPV)
        
        info_before = await self.engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multiPV)
        
        board_copy = board.copy()
        board_copy.push(move)
        
        info_after = await self.engine.analyse(board_copy, chess.engine.Limit(depth=depth), multipv=multiPV)
        
        return {
            "move": move.uci(),
            "evaluation_before": self._format_evaluation(info_before[0]["score"]),
            "evaluation_after": self._format_evaluation(info_after[0]["score"]),
            "best_moves_before": [
                {
                    "move": pv[0].uci() if pv.get("pv") else None,
                    "evaluation": self._format_evaluation(pv["score"]),
                    "pv": [m.uci() for m in pv.get("pv", [])]
                }
                for pv in info_before
            ],
            "best_moves_after": [
                {
                    "move": pv[0].uci() if pv.get("pv") else None,
                    "evaluation": self._format_evaluation(pv["score"]),
                    "pv": [m.uci() for m in pv.get("pv", [])]
                }
                for pv in info_after
            ]
        }
    
    async def get_best_move(self, board: chess.Board, time_limit: float = 1.0, depth: int = None):
        if isinstance(self.engine, MockStockfishEngine):
            return await self.engine.get_best_move(board, time_limit, depth)
        
        limit = chess.engine.Limit(time=time_limit) if depth is None else chess.engine.Limit(depth=depth)
        result = await self.engine.play(board, limit)
        
        info = await self.engine.analyse(board, chess.engine.Limit(depth=15))
        
        return {
            "move": result.move.uci(),
            "evaluation": self._format_evaluation(info["score"]),
            "pv": [m.uci() for m in info.get("pv", [])],
            "depth": info.get("depth", 0)
        }
    
    async def evaluate_position(self, board: chess.Board, depth: int = 15):
        if isinstance(self.engine, MockStockfishEngine):
            return await self.engine.evaluate_position(board, depth)
        
        info = await self.engine.analyse(board, chess.engine.Limit(depth=depth), multipv=5)
        
        return {
            "evaluation": self._format_evaluation(info[0]["score"]),
            "best_moves": [
                {
                    "move": pv[0].uci() if pv.get("pv") else None,
                    "evaluation": self._format_evaluation(pv["score"]),
                    "pv": [m.uci() for m in pv.get("pv", [])],
                    "depth": pv.get("depth", 0)
                }
                for pv in info
            ],
            "position_features": self._analyze_position_features(board)
        }
    
    def _format_evaluation(self, score):
        if score.is_mate():
            return {"type": "mate", "value": score.mate()}
        else:
            return {"type": "cp", "value": score.score()}
    
    def _analyze_position_features(self, board: chess.Board):
        return {
            "material_balance": self._calculate_material_balance(board),
            "king_safety": self._evaluate_king_safety(board),
            "tactical_motifs": self._find_tactical_motifs(board)
        }
    
    def _calculate_material_balance(self, board: chess.Board):
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = sum(piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE)
        black_material = sum(piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK)
        
        return white_material - black_material
    
    def _evaluate_king_safety(self, board: chess.Board):
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        return {
            "white_king_safety": len(list(board.attackers(chess.BLACK, white_king))),
            "black_king_safety": len(list(board.attackers(chess.WHITE, black_king)))
        }
    
    def _find_tactical_motifs(self, board: chess.Board):
        motifs = []
        
        if board.is_check():
            motifs.append("check")
        
        for move in list(board.legal_moves)[:10]:  # Limit to avoid timeout
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                motifs.append("checkmate_threat")
                break
            elif len(list(board_copy.checkers())) > 0:
                motifs.append("check_threat")
        
        return motifs

    async def close(self):
        try:
            if self.engine and not isinstance(self.engine, MockStockfishEngine):
                # Check if engine transport is still alive before trying to quit
                if hasattr(self.engine, 'transport') and not self.engine.transport.is_closing():
                    await self.engine.quit()
                elif hasattr(self.engine, 'process') and self.engine.process and self.engine.process.returncode is None:
                    await self.engine.quit()
        except (RuntimeError, AttributeError) as e:
            print(f"Engine already closed or unavailable: {e}")
        except Exception as e:
            print(f"Error closing engine: {e}")


# Mock Stockfish Engine for Development/Testing
class MockStockfishEngine:
    """Mock engine when Stockfish is not available"""
    
    async def analyze_move(self, board: chess.Board, move: chess.Move, depth: int = 15, multiPV: int = 3):
        # Simple random evaluation for demo purposes
        import random
        
        eval_before = random.randint(-100, 100)
        eval_after = random.randint(-100, 100)
        
        legal_moves = list(board.legal_moves)
        best_moves = []
        
        for i in range(min(multiPV, len(legal_moves))):
            if i < len(legal_moves):
                best_moves.append({
                    "move": legal_moves[i].uci(),
                    "evaluation": {"type": "cp", "value": random.randint(-50, 50)},
                    "pv": [legal_moves[i].uci()]
                })
        
        return {
            "move": move.uci(),
            "evaluation_before": {"type": "cp", "value": eval_before},
            "evaluation_after": {"type": "cp", "value": eval_after},
            "best_moves_before": best_moves,
            "best_moves_after": best_moves
        }
    
    async def get_best_move(self, board: chess.Board, time_limit: float = 1.0, depth: int = None):
        import random
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            raise Exception("No legal moves available")
        
        best_move = random.choice(legal_moves)
        
        return {
            "move": best_move.uci(),
            "evaluation": {"type": "cp", "value": random.randint(-50, 50)},
            "pv": [best_move.uci()],
            "depth": depth or 10
        }
    
    async def evaluate_position(self, board: chess.Board, depth: int = 15):
        import random
        legal_moves = list(board.legal_moves)
        
        best_moves = []
        for i, move in enumerate(legal_moves[:5]):  # Top 5 moves
            best_moves.append({
                "move": move.uci(),
                "evaluation": {"type": "cp", "value": random.randint(-100, 100)},
                "pv": [move.uci()],
                "depth": depth
            })
        
        return {
            "evaluation": {"type": "cp", "value": random.randint(-50, 50)},
            "best_moves": best_moves,
            "position_features": {
                "material_balance": 0,
                "king_safety": {"white_king_safety": 0, "black_king_safety": 0},
                "tactical_motifs": []
            }
        }

    
    def _format_evaluation(self, score):
        if score.is_mate():
            return {"type": "mate", "value": score.mate()}
        else:
            return {"type": "cp", "value": score.score()}
    
    def _analyze_position_features(self, board: chess.Board):
        return {
            "material_balance": self._calculate_material_balance(board),
            "king_safety": self._evaluate_king_safety(board),
            "tactical_motifs": self._find_tactical_motifs(board)
        }
    
    def _calculate_material_balance(self, board: chess.Board):
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = sum(piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE)
        black_material = sum(piece_values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK)
        
        return white_material - black_material
    
    def _evaluate_king_safety(self, board: chess.Board):
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        return {
            "white_king_safety": len(list(board.attackers(chess.BLACK, white_king))),
            "black_king_safety": len(list(board.attackers(chess.WHITE, black_king)))
        }
    
    def _find_tactical_motifs(self, board: chess.Board):
        motifs = []
        
        if board.is_check():
            motifs.append("check")
        
        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                motifs.append("checkmate_threat")
            elif len(list(board_copy.checkers())) > 0:
                motifs.append("check_threat")
        
        return motifs

    async def close(self):
    # Mock engine doesn't need actual cleanup
     pass

# Simple AI Services (mock implementations to avoid missing imports)
class SimpleChessAnalyzer:
    async def initialize(self):
        pass
    
    async def analyze_position(self, fen: str, depth: int = 15):
        board = chess.Board(fen)
        return {
            "fen": fen,
            "legal_moves": [move.uci() for move in board.legal_moves],
            "game_phase": "middlegame",
            "material_balance": 0
        }

class SimpleAICoach:
    async def initialize(self):
        pass
    
    async def provide_advice(self, fen: str, player_level: str = "intermediate", focus_areas: List[str] = []):
        return {
            "general_advice": "Focus on piece development and center control",
            "specific_suggestions": ["Develop knights before bishops", "Castle early for king safety"],
            "focus_areas": focus_areas or ["tactics", "positional play"]
        }
    
    async def classify_move(self, fen: str, move: str, analysis: dict):
        return "good"
    
    async def analyze_position_features(self, fen: str, stockfish_analysis: dict):
        return {"insights": "Position looks balanced"}

class SimplePuzzleGenerator:
    async def generate_puzzles(self, difficulty: str = "intermediate", theme: str = "tactics", count: int = 5):
        return [{"id": f"puzzle_{i}", "fen": chess.STARTING_FEN, "theme": theme} for i in range(count)]

class SimpleSkillAssessor:
    async def load_model(self):
        pass
    
    async def assess_player(self, games: List[Dict], time_controls: List[str] = []):
        return {"estimated_rating": 1500, "strengths": ["tactics"], "weaknesses": ["endgames"]}

class SimpleOpeningCoach:
    async def analyze_opening(self, moves: List[str], color: str = "white"):
        return {
            "opening_name": "Italian Game",
            "evaluation": "Good opening choice",
            "suggestions": ["Continue development"]
        }

# Initialize services
stockfish = StockfishManager()
chess_analyzer = SimpleChessAnalyzer()
ai_coach = SimpleAICoach()
puzzle_generator_original = SimplePuzzleGenerator()
skill_assessor = SimpleSkillAssessor()
opening_coach = SimpleOpeningCoach()
user_manager = UserManager()
enhanced_puzzle_generator = EnhancedPuzzleGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("Chess app starting up...")
    try:
        await stockfish.initialize()
        await chess_analyzer.initialize()
        await ai_coach.initialize()
        await skill_assessor.load_model()
        print("Chess AI Backend initialized successfully!")
        yield
    finally:
        # Shutdown code
        print("Chess app shutting down...")
        try:
            await stockfish.close()
            print("Stockfish engine closed successfully")
        except Exception as e:
            print(f"Error during shutdown: {e}")
            # Continue shutdown process despite error

# Create FastAPI app with lifespan
app = FastAPI(title="Chess AI Coach Backend", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User Management Endpoints
@app.post("/users/register")
async def register_user(request: UserRegistration):
    try:
        result = user_manager.create_user(request.username, request.email)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/stats")
async def get_user_statistics(user_id: str):
    try:
        stats = user_manager.get_user_stats(user_id)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Puzzle Endpoints
@app.post("/puzzles/generate")
async def generate_puzzles_new(request: PuzzleRequest):
    try:
        puzzles = enhanced_puzzle_generator.generate_puzzles(
            request.theme, request.count, request.min_rating, request.max_rating
        )
        return {"puzzles": puzzles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/puzzles/solve")
async def solve_puzzle(request: PuzzleSolution):
    try:
        result = enhanced_puzzle_generator.validate_solution(request.puzzle_id, request.moves)
        
        user_id = "placeholder_user"
        enhanced_puzzle_generator.record_attempt(
            request.puzzle_id, user_id, result.get('complete', False), 
            request.time_taken, request.moves
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Chess Analysis Endpoints
@app.post("/analyze/move")
async def analyze_move_detailed(request: MoveAnalysisRequest):
    try:
        board = chess.Board(request.fen)
        move = chess.Move.from_uci(request.move)
        
        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")
        
        analysis = await stockfish.analyze_move(board, move, request.depth, request.multiPV)
        
        analysis["move_classification"] = await ai_coach.classify_move(
            fen=request.fen,
            move=request.move,
            analysis=analysis
        )
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/position")
async def analyze_position(request: PositionAnalysisRequest):
    try:
        board = chess.Board(request.fen)
        
        analysis = await stockfish.evaluate_position(board, request.depth)
        
        analysis["coach_insights"] = await ai_coach.analyze_position_features(
            fen=request.fen,
            stockfish_analysis=analysis
        )
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Computer Opponent Endpoints
@app.post("/game/computer-move")
async def get_computer_move(request: GameMoveRequest):
    try:
        board = chess.Board(request.fen)
        
        # Adjust engine strength based on difficulty
        time_limit = request.time_limit
        depth = None
        
        difficulty_settings = {
            "beginner": (0.1, 5),
            "intermediate": (0.5, 10),
            "advanced": (1.0, 15),
            "expert": (2.0, 20)
        }
        
        time_limit, depth = difficulty_settings.get(request.difficulty, (1.0, 15))
        
        computer_move = await stockfish.get_best_move(board, time_limit, depth)
        
        return {
            "move": computer_move["move"],
            "evaluation": computer_move["evaluation"],
            "difficulty": request.difficulty,
            "thinking_time": time_limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Game Session Management
@app.post("/game/start-session")
async def start_game_session(request: GameSessionRequest):
    try:
        session_data = {
            "session_id": request.session_id,
            "player_color": request.player_color,
            "difficulty": request.difficulty,
            "tutoring_enabled": request.tutoring_enabled,
            "board": chess.Board(),
            "move_history": [],
            "created_at": datetime.now().isoformat()
        }
        
        active_games[request.session_id] = session_data
        
        response = {"session_created": True, "starting_fen": chess.STARTING_FEN}
        
        if request.player_color == "black":
            computer_move = await stockfish.get_best_move(session_data["board"], 1.0)
            session_data["board"].push(chess.Move.from_uci(computer_move["move"]))
            session_data["move_history"].append({
                "move": computer_move["move"],
                "by": "computer",
                "fen_after": session_data["board"].fen()
            })
            response["computer_first_move"] = computer_move["move"]
            response["fen_after_computer_move"] = session_data["board"].fen()
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/game/make-move")
async def make_move(request: GameMoveSubmission):
    try:
        if request.session_id not in active_games:
            raise HTTPException(status_code=404, detail="Game session not found")
        
        session = active_games[request.session_id]
        board = session["board"]
        
        player_move = chess.Move.from_uci(request.move)
        if player_move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")
        
        move_analysis = None
        if session["tutoring_enabled"]:
            move_analysis = await analyze_player_move_for_tutoring(
                request.fen_before, request.move, session["difficulty"]
            )
        
        board.push(player_move)
        session["move_history"].append({
            "move": request.move,
            "by": "player",
            "fen_after": board.fen(),
            "analysis": move_analysis
        })
        
        game_status = get_game_status(board)
        response = {
            "player_move_accepted": True,
            "fen_after_player_move": board.fen(),
            "game_status": game_status,
            "move_analysis": move_analysis
        }
        
        if game_status["status"] == "ongoing":
            computer_move = await stockfish.get_best_move(board, 1.0)
            board.push(chess.Move.from_uci(computer_move["move"]))
            session["move_history"].append({
                "move": computer_move["move"],
                "by": "computer",
                "fen_after": board.fen()
            })
            
            response["computer_move"] = computer_move["move"]
            response["fen_after_computer_move"] = board.fen()
            response["final_game_status"] = get_game_status(board)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/game/{session_id}/status")
async def get_game_status_endpoint(session_id: str):
    try:
        if session_id not in active_games:
            raise HTTPException(status_code=404, detail="Game session not found")
        
        session = active_games[session_id]
        board = session["board"]
        
        return {
            "session_id": session_id,
            "current_fen": board.fen(),
            "game_status": get_game_status(board),
            "move_count": len(session["move_history"]),
            "last_moves": session["move_history"][-5:] if session["move_history"] else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Coaching and Analysis Endpoints
@app.post("/coaching/position-advice")
async def get_position_advice(request: CoachingRequest):
    try:
        advice = await ai_coach.provide_advice(
            request.fen, request.player_level, request.focus_areas
        )
        
        # Add Stockfish analysis
        board = chess.Board(request.fen)
        stockfish_analysis = await stockfish.evaluate_position(board, depth=12)
        
        return {
            "coaching_advice": advice,
            "position_analysis": stockfish_analysis,
            "recommendations": generate_coaching_recommendations(request.fen, advice, stockfish_analysis)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/skill-assessment")
async def assess_skill(request: SkillAssessmentRequest):
    try:
        assessment = await skill_assessor.assess_player(request.games, request.time_controls)
        return assessment
    except Exception as e:
        raise HTTP
@app.post("/analysis/skill-assessment")
async def assess_skill(request: SkillAssessmentRequest):
    try:
        assessment = await skill_assessor.assess_player(request.games, request.time_controls)
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/opening/analyze")
async def analyze_opening(request: OpeningAnalysisRequest):
    try:
        analysis = await opening_coach.analyze_opening(request.moves, request.color)
        
        # Build position from moves
        board = chess.Board()
        for move_str in request.moves:
            try:
                move = chess.Move.from_uci(move_str)
                board.push(move)
            except:
                # Try SAN notation
                move = board.parse_san(move_str)
                board.push(move)
        
        # Get engine evaluation of resulting position
        engine_analysis = await stockfish.evaluate_position(board, depth=12)
        
        return {
            "opening_analysis": analysis,
            "current_position": {
                "fen": board.fen(),
                "evaluation": engine_analysis["evaluation"],
                "best_continuation": engine_analysis["best_moves"][0] if engine_analysis["best_moves"] else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Tutoring System Endpoints
@app.post("/tutor/analyze-move")
async def tutor_analyze_move(request: TutorAnalysisRequest):
    try:
        analysis = await analyze_player_move_for_tutoring(
            request.fen, request.player_move, request.player_level
        )
        
        if request.explain_alternatives:
            board = chess.Board(request.fen)
            engine_analysis = await stockfish.evaluate_position(board, depth=15)
            
            alternatives = []
            for i, best_move in enumerate(engine_analysis["best_moves"][:3]):
                if best_move["move"] and best_move["move"] != request.player_move:
                    alternatives.append({
                        "move": best_move["move"],
                        "evaluation": best_move["evaluation"],
                        "explanation": generate_move_explanation(best_move["move"], board)
                    })
            
            analysis["alternatives"] = alternatives
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check and Info Endpoints
@app.get("/health")
async def health_check():
    try:
        # Test engine connection
        test_board = chess.Board()
        await stockfish.get_best_move(test_board, time_limit=0.1)
        
        return {
            "status": "healthy",
            "services": {
                "stockfish": "operational",
                "database": "operational",
                "ai_services": "operational"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/info")
async def get_app_info():
    return {
        "name": "Chess AI Coach Backend",
        "version": "1.0.0",
        "features": [
            "Move Analysis",
            "Position Evaluation", 
            "Computer Opponent",
            "Puzzle Generation",
            "Skill Assessment",
            "Opening Analysis",
            "AI Tutoring"
        ],
        "active_games": len(active_games),
        "supported_time_controls": ["bullet", "blitz", "rapid", "classical"]
    }

# Utility Functions
def get_game_status(board: chess.Board):
    """Determine the current game status"""
    if board.is_checkmate():
        winner = "white" if board.turn == chess.BLACK else "black"
        return {"status": "checkmate", "winner": winner, "result": "1-0" if winner == "white" else "0-1"}
    elif board.is_stalemate():
        return {"status": "stalemate", "winner": None, "result": "1/2-1/2"}
    elif board.is_insufficient_material():
        return {"status": "insufficient_material", "winner": None, "result": "1/2-1/2"}
    elif board.is_seventyfive_moves():
        return {"status": "75_move_rule", "winner": None, "result": "1/2-1/2"}
    elif board.is_fivefold_repetition():
        return {"status": "repetition", "winner": None, "result": "1/2-1/2"}
    elif board.can_claim_draw():
        return {"status": "draw_claimable", "winner": None, "result": "1/2-1/2"}
    elif board.is_check():
        return {"status": "check", "winner": None, "result": "*"}
    else:
        return {"status": "ongoing", "winner": None, "result": "*"}

async def analyze_player_move_for_tutoring(fen: str, move: str, player_level: str = "intermediate"):
    """Analyze a player's move and provide educational feedback"""
    try:
        board = chess.Board(fen)
        player_move = chess.Move.from_uci(move)
        
        if player_move not in board.legal_moves:
            return {"error": "Illegal move"}
        
        # Get engine analysis before and after the move
        analysis = await stockfish.analyze_move(board, player_move, depth=15, multiPV=3)
        
        # Classify the move quality
        move_classification = classify_move_quality(analysis)
        
        # Generate educational feedback
        feedback = generate_educational_feedback(analysis, move_classification, player_level)
        
        return {
            "move": move,
            "classification": move_classification,
            "feedback": feedback,
            "engine_analysis": analysis,
            "suggestions": generate_improvement_suggestions(analysis, player_level)
        }
    except Exception as e:
        return {"error": str(e)}

def classify_move_quality(analysis):
    """Classify move quality based on evaluation change"""
    try:
        eval_before = analysis["evaluation_before"]
        eval_after = analysis["evaluation_after"]
        
        if eval_before["type"] == "cp" and eval_after["type"] == "cp":
            diff = eval_after["value"] - eval_before["value"]
            
            if diff >= 50:
                return "excellent"
            elif diff >= 0:
                return "good"
            elif diff >= -25:
                return "inaccurate"
            elif diff >= -100:
                return "mistake"
            else:
                return "blunder"
        
        return "neutral"
    except:
        return "unknown"

def generate_educational_feedback(analysis, classification, player_level):
    """Generate educational feedback based on move analysis"""
    feedback_templates = {
        "excellent": "Great move! This significantly improves your position.",
        "good": "Solid choice. This move maintains your advantage.",
        "inaccurate": "This move is playable but not the most accurate.",
        "mistake": "This move gives your opponent an advantage. Consider the alternatives.",
        "blunder": "This is a serious error that significantly worsens your position."
    }
    
    base_feedback = feedback_templates.get(classification, "Move played.")
    
    # Add level-appropriate details
    if player_level in ["advanced", "expert"]:
        try:
            best_move = analysis["best_moves_before"][0]
            if best_move and best_move.get("move"):
                base_feedback += f" The engine suggests {best_move['move']} as the best continuation."
        except:
            pass
    
    return base_feedback

def generate_improvement_suggestions(analysis, player_level):
    """Generate improvement suggestions based on analysis"""
    suggestions = []
    
    try:
        if analysis.get("best_moves_before"):
            best_move = analysis["best_moves_before"][0]
            suggestions.append(f"Consider {best_move['move']} which the engine evaluates as stronger.")
    except:
        pass
    
    # Add general suggestions based on player level
    level_suggestions = {
        "beginner": [
            "Focus on piece safety and development",
            "Look for basic tactical patterns",
            "Ensure king safety through castling"
        ],
        "intermediate": [
            "Analyze candidate moves more deeply",
            "Consider long-term positional factors",
            "Look for tactical combinations"
        ],
        "advanced": [
            "Evaluate pawn structure implications",
            "Consider dynamic factors over static ones",
            "Calculate variations more precisely"
        ]
    }
    
    suggestions.extend(level_suggestions.get(player_level, level_suggestions["intermediate"]))
    
    return suggestions[:3]  # Limit to 3 suggestions

def generate_coaching_recommendations(fen, coaching_advice, stockfish_analysis):
    """Generate specific coaching recommendations"""
    recommendations = []
    
    # Add recommendations based on position analysis
    try:
        if stockfish_analysis.get("position_features"):
            features = stockfish_analysis["position_features"]
            
            if features.get("material_balance", 0) > 2:
                recommendations.append("You have a material advantage - look for simplifications")
            elif features.get("material_balance", 0) < -2:
                recommendations.append("You're behind in material - seek tactical complications")
            
            king_safety = features.get("king_safety", {})
            if king_safety.get("white_king_safety", 0) > 2:
                recommendations.append("White king looks vulnerable - consider attacking moves")
            if king_safety.get("black_king_safety", 0) > 2:
                recommendations.append("Black king looks vulnerable - consider attacking moves")
    except:
        pass
    
    # Add general recommendations
    recommendations.extend([
        "Always check for tactical opportunities before making a move",
        "Consider your opponent's threats and plans",
        "Improve your worst-placed piece"
    ])
    
    return recommendations[:5]  # Limit to 5 recommendations

def generate_move_explanation(move_uci, board):
    """Generate a human-readable explanation of a move"""
    try:
        move = chess.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        
        if not piece:
            return "Move explanation unavailable"
        
        piece_name = chess.piece_name(piece.piece_type).title()
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        
        explanation = f"{piece_name} from {from_square} to {to_square}"
        
        # Add special move details
        if move.promotion:
            explanation += f" (promotes to {chess.piece_name(move.promotion).title()})"
        elif board.is_castling(move):
            explanation += " (castling)"
        elif board.is_capture(move):
            explanation += " (capture)"
        elif board.is_en_passant(move):
            explanation += " (en passant capture)"
        
        return explanation
    except:
        return "Move explanation unavailable"

# Background Tasks
async def cleanup_old_games():
    """Clean up old game sessions"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session_data in active_games.items():
        try:
            created_at = datetime.fromisoformat(session_data["created_at"])
            if (current_time - created_at).total_seconds() > 3600:  # 1 hour timeout
                sessions_to_remove.append(session_id)
        except:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del active_games[session_id]
    
    print(f"Cleaned up {len(sessions_to_remove)} old game sessions")

# Scheduled cleanup (would need a proper scheduler in production)
@app.on_event("startup")
async def schedule_cleanup():
    # In production, use a proper task scheduler like Celery or APScheduler
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)