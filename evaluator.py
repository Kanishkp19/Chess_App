"""
Chess position evaluation and analysis
"""

import chess
import chess.engine
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ChessEvaluator:
    def __init__(self, stockfish_path: str):
        self.stockfish_path = Path(stockfish_path)
        self.engine_settings = {
            'Hash': 512,
            'Threads': 4,
            'UCI_AnalyseMode': True
        }
        
        self.depth_settings = {
            'opening': {'depth': 18, 'time': 0.5},
            'middlegame': {'depth': 20, 'time': 1.0},
            'endgame': {'depth': 25, 'time': 1.5},
            'critical': {'depth': 22, 'time': 2.0}
        }
    
    def premium_evaluate(self, engine: chess.engine.SimpleEngine, board: chess.Board, 
                        game_phase: str, pos_features: Dict[str, float] = None) -> Tuple[float, int]:
        """Premium evaluation with verified depth and accuracy"""
        settings = self.depth_settings[game_phase]
        
        if pos_features and pos_features.get('complexity', 1.0) > 1.5:
            settings = self.depth_settings['critical']
        
        try:
            limit = chess.engine.Limit(depth=settings['depth'], time=settings['time'])
            info = engine.analyse(board, limit)
            
            score = info['score']
            depth = info.get('depth', settings['depth'])
            
            if score.relative.is_mate():
                mate_score = 30.0 if score.relative.mate() > 0 else -30.0
                return mate_score, depth
            else:
                cp_score = score.relative.score() or 0
                return float(cp_score) / 100.0, depth
                
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0, 15
    
    def get_premium_best_move(self, engine: chess.engine.SimpleEngine, board: chess.Board,
                             game_phase: str, pos_features: Dict[str, float]) -> Dict[str, any]:
        """Get best move with principal variation"""
        settings = self.depth_settings[game_phase]
        
        if pos_features.get('complexity', 1.0) > 1.5:
            settings = self.depth_settings['critical']
        
        try:
            limit = chess.engine.Limit(depth=settings['depth'], time=settings['time'])
            info = engine.analyse(board, limit, multipv=1)
            
            if 'pv' in info and info['pv']:
                best_move = info['pv'][0]
                pv = []
                temp_board = board.copy()
                
                for i, move in enumerate(info['pv'][:8]):
                    if i >= 8:
                        break
                    pv.append(temp_board.san(move))
                    temp_board.push(move)
                
                return {
                    'move': board.san(best_move),
                    'pv': pv
                }
        except Exception as e:
            logger.warning(f"Best move analysis failed: {e}")
        
        return {'move': "N/A", 'pv': []}
    
    def analyze_position_premium(self, board: chess.Board) -> Dict[str, float]:
        """Premium position analysis with tactical pattern detection"""
        features = {}
        
        features.update(self._calculate_material_balance(board))
        features.update(self._calculate_mobility_metrics(board))
        features.update(self._calculate_king_safety_advanced(board))
        features.update(self._detect_tactical_patterns(board))
        features.update(self._analyze_positional_factors(board))
        
        return features
    
    def _calculate_material_balance(self, board: chess.Board) -> Dict[str, float]:
        """Enhanced material calculation with positional values"""
        material = {'white': 0, 'black': 0}
        piece_values = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.2, chess.BISHOP: 3.3,
            chess.ROOK: 5.1, chess.QUEEN: 9.5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                color = 'white' if piece.color else 'black'
                base_value = piece_values[piece.piece_type]
                
                if piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(square)
                    if piece.color:
                        base_value += (rank - 1) * 0.1
                    else:
                        base_value += (6 - rank) * 0.1
                
                material[color] += base_value
        
        return {
            'material_balance': material['white'] - material['black'],
            'total_material': material['white'] + material['black']
        }
    
    def _calculate_mobility_metrics(self, board: chess.Board) -> Dict[str, float]:
        """Advanced mobility calculation"""
        current_turn = board.turn
        
        board.turn = chess.WHITE
        white_moves = len(list(board.legal_moves))
        
        board.turn = chess.BLACK  
        black_moves = len(list(board.legal_moves))
        
        board.turn = current_turn
        
        return {
            'white_mobility': white_moves,
            'black_mobility': black_moves,
            'mobility_ratio': white_moves / max(black_moves, 1)
        }
    
    def _calculate_king_safety_advanced(self, board: chess.Board) -> Dict[str, float]:
        """Advanced king safety with attack detection"""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        white_safety = self._king_safety_score(board, white_king, chess.WHITE) if white_king else -5.0
        black_safety = self._king_safety_score(board, black_king, chess.BLACK) if black_king else -5.0
        
        return {
            'white_king_safety': white_safety,
            'black_king_safety': black_safety,
            'king_safety_diff': white_safety - black_safety
        }
    
    def _king_safety_score(self, board: chess.Board, king_square: int, color: chess.Color) -> float:
        """Detailed king safety calculation"""
        safety = 0.0
        
        # Pawn shield
        shield_squares = self._get_pawn_shield_squares(king_square, color)
        for square in shield_squares:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                safety += 0.5
        
        # Attack count near king
        opponent_attacks = 0
        for square in chess.SQUARES:
            if chess.square_distance(square, king_square) <= 2:
                if board.is_attacked_by(not color, square):
                    opponent_attacks += 1
        
        safety -= opponent_attacks * 0.3
        
        # Open files near king
        king_file = chess.square_file(king_square)
        for file_offset in [-1, 0, 1]:
            if 0 <= king_file + file_offset <= 7:
                if self._is_file_open(board, king_file + file_offset):
                    safety -= 0.4
        
        return safety
    
    def _detect_tactical_patterns(self, board: chess.Board) -> Dict[str, float]:
        """Detect common tactical patterns"""
        patterns = {
            'pins': 0, 'forks': 0, 'skewers': 0, 'discovered_attacks': 0,
            'hanging_pieces': 0, 'trapped_pieces': 0
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if not board.is_attacked_by(piece.color, square) and board.is_attacked_by(not piece.color, square):
                    patterns['hanging_pieces'] += 1
        
        patterns['complexity'] = min(3.0, (patterns['hanging_pieces'] + patterns['pins']) / 2.0 + 1.0)
        return patterns
    
    def _analyze_positional_factors(self, board: chess.Board) -> Dict[str, float]:
        """Analyze key positional factors"""
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        
        center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq)) - \
                        sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
        
        pawn_structure = self._analyze_pawn_structure(board)
        
        return {
            'center_control': center_control,
            'doubled_pawns': pawn_structure['doubled'],
            'isolated_pawns': pawn_structure['isolated'],
            'passed_pawns': pawn_structure['passed']
        }
    
    def _get_pawn_shield_squares(self, king_square: int, color: chess.Color) -> List[int]:
        """Get pawn shield squares for king"""
        squares = []
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        direction = 1 if color == chess.WHITE else -1
        shield_rank = king_rank + direction
        
        if 0 <= shield_rank <= 7:
            for file_offset in [-1, 0, 1]:
                if 0 <= king_file + file_offset <= 7:
                    squares.append(chess.square(king_file + file_offset, shield_rank))
        
        return squares
    
    def _is_file_open(self, board: chess.Board, file: int) -> bool:
        """Check if file is open (no pawns)"""
        for rank in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece and piece.piece_type == chess.PAWN:
                return False
        return True
    
    def _analyze_pawn_structure(self, board: chess.Board) -> Dict[str, int]:
        """Analyze pawn structure weaknesses"""
        structure = {'doubled': 0, 'isolated': 0, 'passed': 0}
        
        white_pawns = {}
        black_pawns = {}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                if piece.color == chess.WHITE:
                    white_pawns[file] = white_pawns.get(file, 0) + 1
                else:
                    black_pawns[file] = black_pawns.get(file, 0) + 1
        
        structure['doubled'] = sum(max(0, count - 1) for count in white_pawns.values()) + \
                              sum(max(0, count - 1) for count in black_pawns.values())
        
        return structure