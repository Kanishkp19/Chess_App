# services/enhanced_chess_analyzer.py
import asyncio
import chess
import chess.engine
import chess.polyglot
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChessAnalyzer:
    def __init__(self, stockfish_path: str, threads: int = 4, hash_size: int = 512):
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_size = hash_size
        self.engine = None
        self.opening_book = None
        self.tactical_patterns = self._load_tactical_patterns()
        
    async def initialize(self):
        """Initialize the chess engine with optimized settings"""
        try:
            # Configure Stockfish with advanced settings
            self.engine = await chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            
            # Optimize engine settings
            await self.engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_size,
                "Ponder": False,
                "MultiPV": 5,
                "Skill Level": 20,
                "Slow Mover": 100,
                "Move Overhead": 30
            })
            
            # Load opening book if available
            try:
                self.opening_book = chess.polyglot.open_reader("data/openings.bin")
            except FileNotFoundError:
                logger.warning("Opening book not found, continuing without it")
                
            logger.info("Enhanced chess analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chess analyzer: {str(e)}")
            raise
    
    def _load_tactical_patterns(self) -> Dict[str, Any]:
        """Load tactical pattern recognition data"""
        patterns = {
            "pin": {"keywords": ["pin", "pinned", "absolute", "relative"], "score_threshold": 200},
            "fork": {"keywords": ["fork", "double attack"], "score_threshold": 300},
            "skewer": {"keywords": ["skewer", "x-ray"], "score_threshold": 250},
            "discovered": {"keywords": ["discovery", "discovered"], "score_threshold": 400},
            "deflection": {"keywords": ["deflection", "overload"], "score_threshold": 300},
            "decoy": {"keywords": ["decoy", "distraction"], "score_threshold": 250},
            "sacrifice": {"keywords": ["sacrifice", "sac"], "score_threshold": 500}
        }
        return patterns
    
    async def deep_analyze_position(
        self,
        fen: str,
        depth: int = 20,
        time_limit: float = 10.0,
        multi_pv: int = 3,
        include_tactical_patterns: bool = True,
        include_positional_themes: bool = True
    ) -> Dict[str, Any]:
        """
        Perform deep position analysis with multiple variations and patterns
        """
        try:
            board = chess.Board(fen)
            
            # Multi-PV analysis for top moves
            info = await self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=time_limit),
                multipv=multi_pv
            )
            
            # Extract main variations
            variations = []
            for i, pv_info in enumerate(info):
                variation = {
                    "rank": i + 1,
                    "score": self._format_score(pv_info.get("score")),
                    "depth": pv_info.get("depth", 0),
                    "nodes": pv_info.get("nodes", 0),
                    "nps": pv_info.get("nps", 0),
                    "pv": [move.uci() for move in pv_info.get("pv", [])],
                    "san_pv": [board.san(move) for move in pv_info.get("pv", [])],
                    "evaluation": self._evaluate_variation(board, pv_info.get("pv", []))
                }
                variations.append(variation)
            
            # Position characteristics
            position_info = self._analyze_position_characteristics(board)
            
            # Tactical pattern detection
            tactical_patterns = []
            if include_tactical_patterns:
                tactical_patterns = await self._detect_tactical_patterns(board, variations)
            
            # Positional themes
            positional_themes = []
            if include_positional_themes:
                positional_themes = self._analyze_positional_themes(board)
            
            # Opening analysis
            opening_info = self._analyze_opening(board)
            
            analysis = {
                "fen": fen,
                "evaluation": variations[0]["score"] if variations else {"type": "cp", "value": 0},
                "best_move": variations[0]["pv"][0] if variations and variations[0]["pv"] else None,
                "variations": variations,
                "position_info": position_info,
                "tactical_patterns": tactical_patterns,
                "positional_themes": positional_themes,
                "opening_info": opening_info,
                "analysis_time": time_limit,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Deep analysis failed: {str(e)}")
            raise
    
    def _format_score(self, score) -> Dict[str, Any]:
        """Format engine score for consistent output"""
        if score is None:
            return {"type": "cp", "value": 0}
        
        if score.is_mate():
            return {
                "type": "mate",
                "value": score.mate(),
                "description": f"Mate in {abs(score.mate())}"
            }
        else:
            cp_value = score.relative.score()
            return {
                "type": "cp",
                "value": cp_value,
                "description": f"{cp_value/100:.2f}" if cp_value else "0.00"
            }
    
    def _analyze_position_characteristics(self, board: chess.Board) -> Dict[str, Any]:
        """Analyze basic position characteristics"""
        return {
            "material_balance": self._calculate_material_balance(board),
            "piece_activity": self._analyze_piece_activity(board),
            "king_safety": self._analyze_king_safety(board),
            "pawn_structure": self._analyze_pawn_structure(board),
            "space_advantage": self._calculate_space_advantage(board),
            "game_phase": self._determine_game_phase(board)
        }
    
    async def _detect_tactical_patterns(self, board: chess.Board, variations: List[Dict]) -> List[Dict]:
        """Detect tactical patterns in the position"""
        patterns = []
        
        if not variations:
            return patterns
        
        best_move_uci = variations[0]["pv"][0] if variations[0]["pv"] else None
        if not best_move_uci:
            return patterns
        
        # Create a copy to test the move
        test_board = board.copy()
        best_move = chess.Move.from_uci(best_move_uci)
        
        if best_move in test_board.legal_moves:
            # Analyze move for tactical patterns
            move_analysis = self._analyze_move_tactics(test_board, best_move)
            patterns.extend(move_analysis)
        
        return patterns
    
    def _analyze_move_tactics(self, board: chess.Board, move: chess.Move) -> List[Dict]:
        """Analyze a specific move for tactical content"""
        tactics = []
        
        # Check for captures
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                tactics.append({
                    "type": "capture",
                    "description": f"Captures {captured_piece.symbol()}",
                    "value": self._piece_values.get(captured_piece.piece_type, 0)
                })
        
        # Check for checks
        board.push(move)
        if board.is_check():
            tactics.append({
                "type": "check",
                "description": "Gives check",
                "forcing": True
            })
        board.pop()
        
        return tactics
    
    @property
    def _piece_values(self) -> Dict[int, int]:
        """Standard piece values"""
        return {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
    
    def is_healthy(self) -> bool:
        """Check if the analyzer is functioning properly"""
        return self.engine is not None
    
    async def cleanup(self):
        """Clean up resources"""
        if self.engine:
            await self.engine.quit()
        if self.opening_book:
            self.opening_book.close()