# services/puzzle_generator.py - Chess Puzzle Generation and Management
import asyncio
import json
import random
import chess
import chess.engine
import requests
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
from datetime import datetime, timedelta

class PuzzleGenerator:
    def __init__(self):
        self.db_path = "puzzles.db"
        self.lichess_api_base = "https://lichess.org/api"
        self.puzzle_cache = {}
        self.engine = None
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for puzzle storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzles (
                id TEXT PRIMARY KEY,
                fen TEXT NOT NULL,
                solution TEXT NOT NULL,
                theme TEXT NOT NULL,
                rating INTEGER NOT NULL,
                description TEXT,
                popularity INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzle_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_id TEXT,
                user_id TEXT,
                solved BOOLEAN,
                time_taken REAL,
                moves_played TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (puzzle_id) REFERENCES puzzles (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def generate_puzzles(self, theme: str, count: int = 10, 
                             min_rating: int = 1200, max_rating: int = 1800) -> List[Dict[str, Any]]:
        """Generate or fetch puzzles based on theme and difficulty"""
        
        # First, try to get from local database
        local_puzzles = self._get_local_puzzles(theme, count, min_rating, max_rating)
        
        if len(local_puzzles) >= count:
            return local_puzzles[:count]
        
        # Fetch more puzzles from external sources
        needed_count = count - len(local_puzzles)
        
        # Try Lichess API first
        lichess_puzzles = await self._fetch_lichess_puzzles(theme, needed_count, min_rating, max_rating)
        
        # Generate synthetic puzzles if needed
        if len(local_puzzles) + len(lichess_puzzles) < count:
            synthetic_puzzles = await self._generate_synthetic_puzzles(
                theme, count - len(local_puzzles) - len(lichess_puzzles), min_rating, max_rating
            )
            return local_puzzles + lichess_puzzles + synthetic_puzzles
        
        return local_puzzles + lichess_puzzles
    
    def _get_local_puzzles(self, theme: str, count: int, min_rating: int, max_rating: int) -> List[Dict[str, Any]]:
        """Get puzzles from local database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, fen, solution, theme, rating, description 
            FROM puzzles 
            WHERE theme = ? AND rating BETWEEN ? AND ?
            ORDER BY popularity DESC, RANDOM()
            LIMIT ?
        ''', (theme, min_rating, max_rating, count))
        
        puzzles = []
        for row in cursor.fetchall():
            puzzles.append({
                "id": row[0],
                "fen": row[1], 
                "solution": json.loads(row[2]),
                "theme": row[3],
                "rating": row[4],
                "description": row[5] or ""
            })
        
        conn.close()
        return puzzles
    
    async def _fetch_lichess_puzzles(self, theme: str, count: int, 
                                   min_rating: int, max_rating: int) -> List[Dict[str, Any]]:
        """Fetch puzzles from Lichess API"""
        try:
            # Map internal themes to Lichess themes
            lichess_theme_map = {
                "fork": "fork",
                "pin": "pin", 
                "skewer": "skewer",
                "discovery": "discoveredAttack",
                "mate_in_1": "mateIn1",
                "mate_in_2": "mateIn2",
                "mate_in_3": "mateIn3",
                "sacrifice": "sacrifice",
                "deflection": "deflection",
                "endgame": "endgame"
            }
            
            lichess_theme = lichess_theme_map.get(theme, theme)
            
            # Fetch puzzle data
            url = f"{self.lichess_api_base}/puzzle/daily"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # This would be expanded to handle batch puzzle fetching
                # For now, return empty list and rely on synthetic generation
                pass
                
        except Exception as e:
            print(f"Lichess API error: {e}")
        
        return []
    
    async def _generate_synthetic_puzzles(self, theme: str, count: int, 
                                        min_rating: int, max_rating: int) -> List[Dict[str, Any]]:
        """Generate synthetic puzzles using chess engine"""
        puzzles = []
        
        for _ in range(count):
            puzzle = await self._create_themed_puzzle(theme, min_rating, max_rating)
            if puzzle:
                puzzles.append(puzzle)
                # Store in database for future use
                self._store_puzzle(puzzle)
        
        return puzzles
    
    async def _create_themed_puzzle(self, theme: str, min_rating: int, max_rating: int) -> Optional[Dict[str, Any]]:
        """Create a puzzle with specific theme"""
        
        if theme == "mate_in_1":
            return await self._generate_mate_in_n_puzzle(1, min_rating, max_rating)
        elif theme == "mate_in_2":
            return await self._generate_mate_in_n_puzzle(2, min_rating, max_rating)
        elif theme == "fork":
            return await self._generate_tactical_puzzle("fork", min_rating, max_rating)
        elif theme == "pin":
            return await self._generate_tactical_puzzle("pin", min_rating, max_rating)
        else:
            return await self._generate_generic_puzzle(theme, min_rating, max_rating)
    
    async def _generate_mate_in_n_puzzle(self, n: int, min_rating: int, max_rating: int) -> Dict[str, Any]:
        """Generate mate in N puzzle"""
        
        # Predefined mate in 1 positions
        mate_in_1_positions = [
            {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
                "solution": ["Qh5+"],
                "description": "Scholar's mate threat"
            },
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 
                "solution": ["Bxf7#"],
                "description": "Legal's mate pattern"
            },
            {
                "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                "solution": ["g3"],
                "description": "Defending against mate threat"
            }
        ]
        
        if n == 1 and mate_in_1_positions:
            position = random.choice(mate_in_1_positions)
            return {
                "id": f"synthetic_mate1_{random.randint(1000, 9999)}",
                "fen": position["fen"],
                "solution": position["solution"],
                "theme": "mate_in_1",
                "rating": random.randint(min_rating, max_rating),
                "description": position["description"]
            }
    
        # For mate in 2+, generate more complex positions
        return {
            "id": f"synthetic_mate{n}_{random.randint(1000, 9999)}",
            "fen": "r3k2r/ppp2ppp/2n1bn2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
            "solution": ["Bxf7+", "Kf8", "Bb3"],
            "theme": f"mate_in_{n}",
            "rating": random.randint(min_rating, max_rating),
            "description": f"Find mate in {n} moves"
        }
    
    async def _generate_tactical_puzzle(self, tactic: str, min_rating: int, max_rating: int) -> Dict[str, Any]:
        """Generate tactical puzzles (fork, pin, etc.)"""
        
        tactical_positions = {
            "fork": [
                {
                    "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                    "solution": ["Nc3"],
                    "description": "Knight develops with fork threat"
                },
                {
                    "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
                    "solution": ["Nd5"],
                    "description": "Knight fork on king and queen"
                }
            ],
            "pin": [
                {
                    "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",
                    "solution": ["Bg5"],
                    "description": "Pin the knight to the queen"
                },
                {
                    "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                    "solution": ["Bb5"],
                    "description": "Pin the knight to the king"
                }
            ],
            "skewer": [
                {
                    "fen": "r3k2r/ppp2ppp/2n1bn2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
                    "solution": ["Bxf7+"],
                    "description": "Skewer the king and rook"
                }
            ]
        }
        
        if tactic in tactical_positions:
            position = random.choice(tactical_positions[tactic])
            return {
                "id": f"synthetic_{tactic}_{random.randint(1000, 9999)}",
                "fen": position["fen"],
                "solution": position["solution"],
                "theme": tactic,
                "rating": random.randint(min_rating, max_rating),
                "description": position["description"]
            }
        
        # Generic fallback
        return await self._generate_generic_puzzle(tactic, min_rating, max_rating)
    
    async def _generate_generic_puzzle(self, theme: str, min_rating: int, max_rating: int) -> Dict[str, Any]:
        """Generate a generic puzzle when specific patterns aren't available"""
        
        generic_positions = [
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "solution": ["d3"],
                "description": "Develop pieces and control center"
            },
            {
                "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "solution": ["exd5"],
                "description": "Capture in the center"
            },
            {
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
                "solution": ["0-0"],
                "description": "Castle for king safety"
            }
        ]
        
        position = random.choice(generic_positions)
        return {
            "id": f"synthetic_{theme}_{random.randint(1000, 9999)}",
            "fen": position["fen"],
            "solution": position["solution"],
            "theme": theme,
            "rating": random.randint(min_rating, max_rating),
            "description": position["description"]
        }
    
    def _store_puzzle(self, puzzle: Dict[str, Any]):
        """Store puzzle in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO puzzles (id, fen, solution, theme, rating, description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            puzzle["id"],
            puzzle["fen"],
            json.dumps(puzzle["solution"]),
            puzzle["theme"],
            puzzle["rating"],
            puzzle["description"]
        ))
        
        conn.commit()
        conn.close()
    
    def record_attempt(self, puzzle_id: str, user_id: str, solved: bool, 
                      time_taken: float, moves_played: List[str]):
        """Record a puzzle attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO puzzle_attempts (puzzle_id, user_id, solved, time_taken, moves_played)
            VALUES (?, ?, ?, ?, ?)
        ''', (puzzle_id, user_id, solved, time_taken, json.dumps(moves_played)))
        
        # Update puzzle popularity
        if solved:
            cursor.execute('''
                UPDATE puzzles SET popularity = popularity + 1 WHERE id = ?
            ''', (puzzle_id,))
        
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user puzzle statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN solved THEN 1 ELSE 0 END) as solved_count,
                AVG(time_taken) as avg_time,
                COUNT(DISTINCT puzzle_id) as unique_puzzles
            FROM puzzle_attempts 
            WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        
        stats = {
            "total_attempts": row[0] or 0,
            "solved_count": row[1] or 0,
            "solve_rate": (row[1] / row[0] * 100) if row[0] > 0 else 0,
            "avg_time": row[2] or 0,
            "unique_puzzles": row[3] or 0
        }
        
        conn.close()
        return stats
    
    def get_puzzle_by_id(self, puzzle_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific puzzle by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, fen, solution, theme, rating, description 
            FROM puzzles WHERE id = ?
        ''', (puzzle_id,))
        
        row = cursor.fetchone()
        if row:
            puzzle = {
                "id": row[0],
                "fen": row[1],
                "solution": json.loads(row[2]),
                "theme": row[3],
                "rating": row[4],
                "description": row[5] or ""
            }
            conn.close()
            return puzzle
        
        conn.close()
        return None
    
    async def validate_solution(self, puzzle_id: str, moves: List[str]) -> Dict[str, Any]:
        """Validate if the provided moves solve the puzzle"""
        puzzle = self.get_puzzle_by_id(puzzle_id)
        if not puzzle:
            return {"valid": False, "error": "Puzzle not found"}
        
        try:
            board = chess.Board(puzzle["fen"])
            expected_moves = puzzle["solution"]
            
            # Check if moves match expected solution
            if len(moves) < len(expected_moves):
                return {
                    "valid": False, 
                    "partial": True,
                    "next_move": expected_moves[len(moves)] if len(moves) < len(expected_moves) else None
                }
            
            # Validate each move
            for i, move_str in enumerate(moves[:len(expected_moves)]):
                try:
                    move = board.parse_san(move_str)
                    if move_str != expected_moves[i]:
                        return {"valid": False, "error": f"Incorrect move at position {i+1}"}
                    board.push(move)
                except ValueError:
                    return {"valid": False, "error": f"Invalid move: {move_str}"}
            
            return {"valid": True, "complete": len(moves) >= len(expected_moves)}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

# Additional files for the chess app

# models/game.py - Game State Management
class ChessGame:
    def __init__(self, game_id: str, white_player: str, black_player: str = None):
        self.game_id = game_id
        self.white_player = white_player
        self.black_player = black_player
        self.board = chess.Board()
        self.moves_history = []
        self.created_at = datetime.utcnow()
        self.status = "active"  # active, completed, abandoned
        self.result = None  # white_wins, black_wins, draw
        self.time_control = {"white": 600, "black": 600}  # 10 minutes each
        self.last_move_time = datetime.utcnow()
    
    def make_move(self, move_str: str, player: str) -> Dict[str, Any]:
        """Make a move in the game"""
        try:
            # Validate player turn
            current_player = self.white_player if self.board.turn else self.black_player
            if player != current_player:
                return {"success": False, "error": "Not your turn"}
            
            # Parse and validate move
            move = self.board.parse_san(move_str)
            if move not in self.board.legal_moves:
                return {"success": False, "error": "Illegal move"}
            
            # Make the move
            self.board.push(move)
            self.moves_history.append({
                "move": move_str,
                "player": player,
                "timestamp": datetime.utcnow(),
                "fen": self.board.fen()
            })
            
            # Check game end conditions
            if self.board.is_checkmate():
                self.status = "completed"
                self.result = "white_wins" if not self.board.turn else "black_wins"
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                self.status = "completed"
                self.result = "draw"
            
            return {
                "success": True,
                "fen": self.board.fen(),
                "status": self.status,
                "result": self.result,
                "in_check": self.board.is_check()
            }
            
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    def get_legal_moves(self) -> List[str]:
        """Get all legal moves in current position"""
        return [self.board.san(move) for move in self.board.legal_moves]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert game to dictionary for serialization"""
        return {
            "game_id": self.game_id,
            "white_player": self.white_player,
            "black_player": self.black_player,
            "fen": self.board.fen(),
            "moves_history": self.moves_history,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "time_control": self.time_control
        }