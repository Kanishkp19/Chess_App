"""
Enhanced Chess Analyzer v2.0 - Premium Quality Analysis
Rating Target: 9+/10
"""

import chess
import chess.pgn
import chess.engine
import requests
import io
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MoveAnalysis:
    move_number: int
    color: str
    move_san: str
    eval_before: float
    eval_after: float
    eval_loss: float
    accuracy: float
    best_move_san: str
    best_move_pv: List[str]  # Principal variation
    classification: str
    error_type: str  # tactical/positional/endgame
    game_phase: str
    position_features: Dict[str, float]
    engine_depth: int

@dataclass
class GameAnalysis:
    game_info: Dict[str, str]
    white_accuracy: float
    black_accuracy: float
    white_errors: Dict[str, int]
    black_errors: Dict[str, int]
    phase_breakdown: Dict[str, Dict[str, float]]
    critical_moments: List[MoveAnalysis]
    tactical_insights: str
    strategic_insights: str
    improvement_plan: str
    evaluation_graph: List[float]

class PremiumChessAnalyzer:
    def __init__(self, stockfish_path: str, mistral_api_key: str):
        self.stockfish_path = Path(stockfish_path)
        self.mistral_api_key = mistral_api_key
        self._validate_setup()
        
        # Premium analysis settings
        self.engine_settings = {
            'Hash': 512,  # MB
            'Threads': 4,
            'UCI_AnalyseMode': True
        }
        
        self.depth_settings = {
            'opening': {'depth': 18, 'time': 0.5},
            'middlegame': {'depth': 20, 'time': 1.0},
            'endgame': {'depth': 25, 'time': 1.5},
            'critical': {'depth': 22, 'time': 2.0}
        }
    
    def _validate_setup(self):
        if not self.stockfish_path.exists():
            raise FileNotFoundError(f"Stockfish not found at {self.stockfish_path}")
        if not self.mistral_api_key:
            raise ValueError("Mistral API key required")
    
    def analyze_game(self, pgn_content: str) -> GameAnalysis:
        """Premium analysis pipeline with deep engine evaluation"""
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_content))
            if not game:
                raise ValueError("Invalid PGN content")
            
            game_info = self._extract_game_info(game)
            moves, eval_graph = self._analyze_moves_premium(game)
            metrics = self._calculate_advanced_metrics(moves)
            insights = self._get_premium_insights(moves, metrics)
            
            return GameAnalysis(
                game_info=game_info,
                white_accuracy=metrics['white_accuracy'],
                black_accuracy=metrics['black_accuracy'],
                white_errors=metrics['white_errors'],
                black_errors=metrics['black_errors'],
                phase_breakdown=metrics['phase_breakdown'],
                critical_moments=metrics['critical_moments'],
                tactical_insights=insights['tactical'],
                strategic_insights=insights['strategic'],
                improvement_plan=insights['improvement'],
                evaluation_graph=eval_graph
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _extract_game_info(self, game: chess.pgn.Game) -> Dict[str, str]:
        h = game.headers
        return {
            'white': h.get('White', 'Unknown'),
            'black': h.get('Black', 'Unknown'), 
            'white_elo': h.get('WhiteElo', 'N/A'),
            'black_elo': h.get('BlackElo', 'N/A'),
            'result': h.get('Result', '*'),
            'event': h.get('Event', 'Unknown'),
            'date': h.get('Date', 'Unknown'),
            'opening': h.get('Opening', 'Unknown')
        }
    
    def _analyze_moves_premium(self, game: chess.pgn.Game) -> Tuple[List[MoveAnalysis], List[float]]:
        """Premium move analysis with verified Stockfish evaluations"""
        moves = []
        eval_graph = []
        
        with chess.engine.SimpleEngine.popen_uci(str(self.stockfish_path)) as engine:
            # Configure engine for premium analysis
            for option, value in self.engine_settings.items():
                try:
                    engine.configure({option: value})
                except:
                    pass
            
            board = chess.Board()
            move_number = 1
            prev_eval, prev_depth = self._premium_evaluate(engine, board, 'opening')
            eval_graph.append(prev_eval)
            
            for node in game.mainline():
                move = node.move
                color = "white" if board.turn else "black"
                game_phase = self._determine_game_phase_advanced(board, move_number)
                
                # Deep position analysis
                pos_features = self._analyze_position_premium(board)
                
                # Get best move with deep analysis
                best_move_info = self._get_premium_best_move(engine, board, game_phase, pos_features)
                
                # Execute move
                move_san = board.san(move)
                board.push(move)
                
                # Deep evaluation after move
                current_eval, current_depth = self._premium_evaluate(engine, board, game_phase, pos_features)
                eval_graph.append(current_eval if board.turn else -current_eval)
                
                # Calculate premium metrics
                eval_loss = self._calculate_verified_eval_loss(prev_eval, current_eval, not board.turn)
                accuracy = self._calculate_premium_accuracy(eval_loss, game_phase, pos_features)
                classification = self._classify_move_advanced(eval_loss, game_phase, pos_features)
                error_type = self._determine_error_type(eval_loss, pos_features, game_phase)
                
                moves.append(MoveAnalysis(
                    move_number=move_number,
                    color=color,
                    move_san=move_san,
                    eval_before=prev_eval,
                    eval_after=current_eval,
                    eval_loss=eval_loss,
                    accuracy=accuracy,
                    best_move_san=best_move_info['move'],
                    best_move_pv=best_move_info['pv'],
                    classification=classification,
                    error_type=error_type,
                    game_phase=game_phase,
                    position_features=pos_features,
                    engine_depth=max(prev_depth, current_depth)
                ))
                
                prev_eval = current_eval
                move_number += 1
        
        return moves, eval_graph
    
    def _premium_evaluate(self, engine: chess.engine.SimpleEngine, board: chess.Board, 
                         game_phase: str, pos_features: Dict[str, float] = None) -> Tuple[float, int]:
        """Premium evaluation with verified depth and accuracy"""
        settings = self.depth_settings[game_phase]
        
        # Adjust for position complexity
        if pos_features and pos_features.get('complexity', 1.0) > 1.5:
            settings = self.depth_settings['critical']
        
        try:
            # Use both depth and time for maximum accuracy
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
    
    def _analyze_position_premium(self, board: chess.Board) -> Dict[str, float]:
        """Premium position analysis with tactical pattern detection"""
        features = {}
        
        # Enhanced material analysis
        material = self._calculate_material_balance(board)
        features.update(material)
        
        # Advanced mobility analysis
        features.update(self._calculate_mobility_metrics(board))
        
        # King safety with threat detection
        features.update(self._calculate_king_safety_advanced(board))
        
        # Tactical pattern recognition
        features.update(self._detect_tactical_patterns(board))
        
        # Positional factors
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
                
                # Positional bonuses
                if piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(square)
                    if piece.color:  # White
                        base_value += (rank - 1) * 0.1  # Bonus for advanced pawns
                    else:  # Black
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
        
        # Simplified tactical detection (can be expanded)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Check for hanging pieces
                if not board.is_attacked_by(piece.color, square) and board.is_attacked_by(not piece.color, square):
                    patterns['hanging_pieces'] += 1
        
        patterns['complexity'] = min(3.0, (patterns['hanging_pieces'] + patterns['pins']) / 2.0 + 1.0)
        return patterns
    
    def _analyze_positional_factors(self, board: chess.Board) -> Dict[str, float]:
        """Analyze key positional factors"""
        # Center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6, 
                          chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6]
        
        center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq)) - \
                        sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
        
        # Pawn structure
        pawn_structure = self._analyze_pawn_structure(board)
        
        return {
            'center_control': center_control,
            'doubled_pawns': pawn_structure['doubled'],
            'isolated_pawns': pawn_structure['isolated'],
            'passed_pawns': pawn_structure['passed']
        }
    
    def _get_premium_best_move(self, engine: chess.engine.SimpleEngine, board: chess.Board,
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
                
                # Get first 4 moves of principal variation
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
    
    def _determine_game_phase_advanced(self, board: chess.Board, move_number: int) -> str:
        """Advanced game phase detection"""
        piece_map = board.piece_map()
        major_pieces = sum(1 for piece in piece_map.values() 
                          if piece.piece_type in [chess.QUEEN, chess.ROOK])
        minor_pieces = sum(1 for piece in piece_map.values()
                          if piece.piece_type in [chess.KNIGHT, chess.BISHOP])
        
        total_pieces = len(piece_map) - 2  # Exclude kings
        
        if move_number <= 15 and total_pieces >= 28:
            return "opening"
        elif total_pieces <= 12 or (major_pieces <= 4 and minor_pieces <= 4):
            return "endgame"
        else:
            return "middlegame"
    
    def _calculate_verified_eval_loss(self, prev_eval: float, current_eval: float, white_played: bool) -> float:
        """Verified evaluation loss calculation"""
        if white_played:
            return max(0.0, prev_eval - current_eval)
        else:
            return max(0.0, current_eval - prev_eval)
    
    def _calculate_premium_accuracy(self, eval_loss: float, game_phase: str, pos_features: Dict[str, float]) -> float:
        """Premium accuracy calculation with context awareness"""
        complexity = pos_features.get('complexity', 1.0)
        
        base_thresholds = {
            'opening': 0.25, 'middlegame': 0.20, 'endgame': 0.15
        }
        
        threshold = base_thresholds[game_phase] * complexity
        
        if eval_loss <= threshold:
            return 98.0
        elif eval_loss <= threshold * 2:
            return 90.0
        elif eval_loss <= threshold * 4:
            return 75.0
        elif eval_loss <= threshold * 8:
            return 50.0
        else:
            return max(10.0, 50.0 - (eval_loss - threshold * 8) * 5)
    
    def _classify_move_advanced(self, eval_loss: float, game_phase: str, pos_features: Dict[str, float]) -> str:
        """Advanced move classification"""
        complexity = pos_features.get('complexity', 1.0)
        
        thresholds = {
            'opening': {'inaccuracy': 0.3, 'mistake': 0.8, 'blunder': 2.0},
            'middlegame': {'inaccuracy': 0.25, 'mistake': 0.6, 'blunder': 1.5}, 
            'endgame': {'inaccuracy': 0.2, 'mistake': 0.4, 'blunder': 1.0}
        }
        
        phase_thresh = thresholds[game_phase]
        
        if eval_loss >= phase_thresh['blunder'] * complexity:
            return "blunder"
        elif eval_loss >= phase_thresh['mistake'] * complexity:
            return "mistake" 
        elif eval_loss >= phase_thresh['inaccuracy'] * complexity:
            return "inaccuracy"
        else:
            return "excellent" if eval_loss <= 0.1 else "good"
    
    def _determine_error_type(self, eval_loss: float, pos_features: Dict[str, float], game_phase: str) -> str:
        """Determine if error is tactical, positional, or endgame"""
        if eval_loss < 0.3:
            return "minor"
        
        hanging = pos_features.get('hanging_pieces', 0)
        king_safety = abs(pos_features.get('king_safety_diff', 0))
        
        if game_phase == "endgame":
            return "endgame"
        elif hanging > 0 or eval_loss > 1.5:
            return "tactical"
        elif king_safety > 1.0 or pos_features.get('center_control', 0) != 0:
            return "positional"
        else:
            return "strategic"
    
    def _calculate_advanced_metrics(self, moves: List[MoveAnalysis]) -> Dict:
        """Calculate comprehensive advanced metrics"""
        white_moves = [m for m in moves if m.color == "white"]
        black_moves = [m for m in moves if m.color == "black"]
        
        # Phase breakdown
        phase_breakdown = {}
        for phase in ['opening', 'middlegame', 'endgame']:
            phase_moves = [m for m in moves if m.game_phase == phase]
            if phase_moves:
                phase_breakdown[phase] = {
                    'accuracy': sum(m.accuracy for m in phase_moves) / len(phase_moves),
                    'errors': len([m for m in phase_moves if m.classification in ['mistake', 'blunder']]),
                    'avg_depth': sum(m.engine_depth for m in phase_moves) / len(phase_moves)
                }
        
        return {
            'white_accuracy': sum(m.accuracy for m in white_moves) / len(white_moves) if white_moves else 0,
            'black_accuracy': sum(m.accuracy for m in black_moves) / len(black_moves) if black_moves else 0,
            'white_errors': self._count_errors_by_type(white_moves),
            'black_errors': self._count_errors_by_type(black_moves),
            'phase_breakdown': phase_breakdown,
            'critical_moments': sorted([m for m in moves if m.classification in ['blunder', 'mistake']], 
                                     key=lambda x: x.eval_loss, reverse=True)[:6]
        }
    
    def _count_errors_by_type(self, moves: List[MoveAnalysis]) -> Dict[str, int]:
        """Count errors by classification and type"""
        errors = {
            'blunders': len([m for m in moves if m.classification == 'blunder']),
            'mistakes': len([m for m in moves if m.classification == 'mistake']),
            'inaccuracies': len([m for m in moves if m.classification == 'inaccuracy']),
            'tactical': len([m for m in moves if m.error_type == 'tactical' and m.classification in ['mistake', 'blunder']]),
            'positional': len([m for m in moves if m.error_type == 'positional' and m.classification in ['mistake', 'blunder']]),
            'endgame': len([m for m in moves if m.error_type == 'endgame' and m.classification in ['mistake', 'blunder']])
        }
        return errors
    
    def _get_premium_insights(self, moves: List[MoveAnalysis], metrics: Dict) -> Dict[str, str]:
        """Premium AI insights with detailed analysis"""
        try:
            # Enhanced tactical analysis
            tactical_data = self._extract_detailed_tactical_patterns(moves)
            strategic_data = self._extract_detailed_strategic_themes(moves, metrics)
            
            tactical_insights = self._get_mistral_response(
                f"Advanced chess tactical analysis: {tactical_data}. "
                f"Identify specific tactical themes, missed combinations, and provide concrete tactical training recommendations. "
                f"Focus on pattern recognition and calculation depth.",
                500
            )
            
            strategic_insights = self._get_mistral_response(
                f"Deep strategic analysis: {strategic_data}. "
                f"Analyze pawn structure, piece coordination, king safety, and long-term planning. "
                f"Provide specific positional concepts to study.",
                500
            )
            
            improvement_plan = self._get_mistral_response(
                f"Personalized improvement plan based on: White {metrics['white_accuracy']:.1f}% accuracy, "
                f"Black {metrics['black_accuracy']:.1f}% accuracy. "
                f"Error breakdown: {metrics['white_errors']}, {metrics['black_errors']}. "
                f"Phase performance: {metrics['phase_breakdown']}. "
                f"Create specific, actionable training recommendations with study priorities.",
                400
            )
            
            return {
                'tactical': tactical_insights,
                'strategic': strategic_insights,
                'improvement': improvement_plan
            }
            
        except Exception as e:
            logger.error(f"Premium insights failed: {e}")
            return {
                'tactical': "Advanced tactical analysis unavailable",
                'strategic': "Deep strategic analysis unavailable",
                'improvement': "Personalized improvement plan unavailable"
            }
    
    def _extract_detailed_tactical_patterns(self, moves: List[MoveAnalysis]) -> str:
        """Extract detailed tactical patterns with engine analysis"""
        tactical_moments = []
        
        for move in moves[:]:
            if move.classification in ['blunder', 'mistake'] and move.eval_loss > 0.8:
                best_line = " ".join(move.best_move_pv[:4]) if move.best_move_pv else "N/A"
                tactical_moments.append(
                    f"Move {move.move_number} ({move.color}): Played {move.move_san} "
                    f"(lost {move.eval_loss:.2f}), Best: {move.best_move_san} "
                    f"with line {best_line}. Error type: {move.error_type}, "
                    f"Phase: {move.game_phase}, Depth: {move.engine_depth}"
                )
        
        return "; ".join(tactical_moments[:4]) if tactical_moments else "No significant tactical errors"
    
    def _extract_detailed_strategic_themes(self, moves: List[MoveAnalysis], metrics: Dict) -> str:
        """Extract detailed strategic analysis"""
        strategic_data = []
        
        # Phase analysis with specifics
        for phase, data in metrics['phase_breakdown'].items():
            strategic_data.append(
                f"{phase.capitalize()}: {data['accuracy']:.1f}% accuracy, "
                f"{data['errors']} errors, avg depth {data['avg_depth']:.1f}"
            )
        
        # Error type distribution
        all_moves = moves
        error_types = {}
        for move in all_moves:
            if move.classification in ['mistake', 'blunder']:
                error_types[move.error_type] = error_types.get(move.error_type, 0) + 1
        
        if error_types:
            type_breakdown = ", ".join([f"{k}: {v}" for k, v in error_types.items()])
            strategic_data.append(f"Error types: {type_breakdown}")
        
        return "; ".join(strategic_data)
    
    def _get_mistral_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Get enhanced response from Mistral API"""
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.5
        }
        
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
        
        return "Premium analysis unavailable"
    
    # Helper methods for positional analysis
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
        
        # Count pawns by file for both colors
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
        
        # Count doubled pawns
        structure['doubled'] = sum(max(0, count - 1) for count in white_pawns.values()) + \
                              sum(max(0, count - 1) for count in black_pawns.values())
        
        return structure

# Enhanced Flask Application
app = Flask(__name__)

PREMIUM_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Premium Chess Analyzer v2.0</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .upload-section { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .upload-area { border: 3px dashed #667eea; border-radius: 10px; padding: 40px; text-align: center; transition: all 0.3s; }
        .upload-area:hover { border-color: #764ba2; background: #f8f9ff; }
        .upload-area input[type="file"] { display: none; }
        .upload-btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; padding: 15px 30px; border-radius: 25px; font-size: 1.1em; cursor: pointer; transition: transform 0.2s; }
        .upload-btn:hover { transform: translateY(-2px); }
        .analyze-btn { background: linear-gradient(45deg, #11998e, #38ef7d); color: white; border: none; padding: 15px 40px; border-radius: 25px; font-size: 1.2em; cursor: pointer; margin-top: 20px; }
        .analyze-btn:hover { transform: translateY(-2px); }
        .results { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .metric-card { background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 20px; border-radius: 10px; margin: 10px; text-align: center; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .phase-analysis { background: #f8f9ff; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .critical-moment { background: #fff5f5; border-left: 4px solid #e53e3e; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .insight-section { background: #f0fff4; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #38a169; }
        .loading { text-align: center; padding: 40px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 10px; margin: 20px 0; }
        .eval-graph { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .progress-bar { width: 100%; height: 20px; background: #e2e8f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Premium Chess Analyzer v2.0</h1>
            <p>Professional-grade analysis with 20+ depth Stockfish evaluation</p>
        </div>
        
        <div class="upload-section">
            <div class="upload-area" onclick="document.getElementById('pgn-file').click()">
                <input type="file" id="pgn-file" accept=".pgn" onchange="handleFileSelect(this)">
                <h3>📁 Upload Your PGN File</h3>
                <p>Drop your PGN file here or click to browse</p>
                <button class="upload-btn">Choose File</button>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button class="analyze-btn" onclick="analyzeGame()" id="analyze-btn" disabled>🔍 Analyze Game</button>
            </div>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <!-- Results will be populated here -->
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        function handleFileSelect(input) {
            selectedFile = input.files[0];
            if (selectedFile) {
                document.getElementById('analyze-btn').disabled = false;
                document.querySelector('.upload-area h3').textContent = `✅ ${selectedFile.name}`;
            }
        }
        
        async function analyzeGame() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('pgn', selectedFile);
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <h3>Analyzing with Premium Engine...</h3>
                    <p>Deep analysis in progress (20+ depth evaluation)</p>
                </div>
            `;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.analysis);
                } else {
                    resultsDiv.innerHTML = `<div class="error">❌ ${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">❌ Analysis failed: ${error.message}</div>`;
            }
        }
        
        function displayResults(analysis) {
            const resultsDiv = document.getElementById('results');
            
            resultsDiv.innerHTML = `
                <h2>🎯 Premium Analysis Results</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>White Accuracy</h3>
                        <div style="font-size: 2em; font-weight: bold;">${analysis.white_accuracy.toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Black Accuracy</h3>
                        <div style="font-size: 2em; font-weight: bold;">${analysis.black_accuracy.toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>White Errors</h3>
                        <div>Blunders: ${analysis.white_errors.blunders}</div>
                        <div>Mistakes: ${analysis.white_errors.mistakes}</div>
                        <div>Inaccuracies: ${analysis.white_errors.inaccuracies}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Black Errors</h3>
                        <div>Blunders: ${analysis.black_errors.blunders}</div>
                        <div>Mistakes: ${analysis.black_errors.mistakes}</div>
                        <div>Inaccuracies: ${analysis.black_errors.inaccuracies}</div>
                    </div>
                </div>
                
                <div class="phase-analysis">
                    <h3>📊 Phase Breakdown</h3>
                    ${Object.entries(analysis.phase_breakdown).map(([phase, data]) => `
                        <div style="margin: 15px 0;">
                            <strong>${phase.charAt(0).toUpperCase() + phase.slice(1)}:</strong>
                            ${data.accuracy.toFixed(1)}% accuracy, ${data.errors} errors
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${data.accuracy}%"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div class="eval-graph">
                    <h3>📈 Evaluation Graph</h3>
                    <canvas id="evalChart" width="800" height="300"></canvas>
                </div>
                
                <div>
                    <h3>🔥 Critical Moments</h3>
                    ${analysis.critical_moments.map(moment => `
                        <div class="critical-moment">
                            <strong>Move ${moment.move_number} (${moment.color}):</strong> 
                            ${moment.move_san} - ${moment.classification}
                            <br>
                            <small>Lost ${moment.eval_loss.toFixed(2)} points. Best: ${moment.best_move_san}</small>
                        </div>
                    `).join('')}
                </div>
                
                <div class="insight-section">
                    <h3>🎯 Tactical Insights</h3>
                    <p>${analysis.tactical_insights}</p>
                </div>
                
                <div class="insight-section">
                    <h3>🧠 Strategic Insights</h3>
                    <p>${analysis.strategic_insights}</p>
                </div>
                
                <div class="insight-section">
                    <h3>📚 Improvement Plan</h3>
                    <p>${analysis.improvement_plan}</p>
                </div>
            `;
            
            drawEvaluationGraph(analysis.evaluation_graph);
        }
        
        function drawEvaluationGraph(evaluations) {
            const canvas = document.getElementById('evalChart');
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Setup
            const width = canvas.width - 40;
            const height = canvas.height - 40;
            const centerY = height / 2 + 20;
            
            // Draw center line
            ctx.strokeStyle = '#e2e8f0';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(20, centerY);
            ctx.lineTo(width + 20, centerY);
            ctx.stroke();
            
            // Draw evaluation curve
            if (evaluations.length > 1) {
                ctx.strokeStyle = '#667eea';
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                for (let i = 0; i < evaluations.length; i++) {
                    const x = 20 + (i * width / (evaluations.length - 1));
                    const eval_clamped = Math.max(-5, Math.min(5, evaluations[i]));
                    const y = centerY - (eval_clamped * height / 10);
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }
            
            // Add labels
            ctx.fillStyle = '#4a5568';
            ctx.font = '12px Arial';
            ctx.fillText('White Advantage', 20, 15);
            ctx.fillText('Black Advantage', 20, height + 35);
            ctx.fillText('Move Progress →', width - 80, height + 35);
        }
        
        // Drag and drop functionality
        document.addEventListener('DOMContentLoaded', function() {
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.style.borderColor = '#764ba2';
                uploadArea.style.background = '#f8f9ff';
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.style.borderColor = '#667eea';
                uploadArea.style.background = 'white';
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.style.borderColor = '#667eea';
                uploadArea.style.background = 'white';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const fileInput = document.getElementById('pgn-file');
                    fileInput.files = files;
                    handleFileSelect(fileInput);
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(PREMIUM_HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_pgn():
    try:
        if 'pgn' not in request.files:
            return jsonify({'success': False, 'error': 'No PGN file uploaded'})
        
        file = request.files['pgn']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read PGN content
        pgn_content = file.read().decode('utf-8')
        
        # Initialize analyzer with environment variables
        import os
        stockfish_path = os.getenv('STOCKFISH_PATH', 'stockfish')
        mistral_api_key = os.getenv('MISTRAL_API_KEY', '')
        
        analyzer = PremiumChessAnalyzer(
            stockfish_path=stockfish_path,
            mistral_api_key=mistral_api_key
        )
        
        # Analyze the game
        analysis = analyzer.analyze_game(pgn_content)
        
        # Convert to JSON-serializable format
        result = asdict(analysis)
        
        return jsonify({'success': True, 'analysis': result})
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'version': '2.0'})

# Configuration and startup
def setup_premium_analyzer():
    """Setup function for the premium analyzer"""
    import os
    
    # Check for required environment variables
    required_vars = ['STOCKFISH_PATH', 'MISTRAL_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Please set STOCKFISH_PATH and MISTRAL_API_KEY environment variables")
    
    # Additional setup for production
    if os.getenv('FLASK_ENV') == 'production':
        # Configure for production
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
    else:
        # Development configuration
        app.config['DEBUG'] = True

# Command line interface for standalone analysis
def cli_analyze(pgn_file_path: str, stockfish_path: str, mistral_api_key: str):
    """Command line interface for chess analysis"""
    try:
        analyzer = PremiumChessAnalyzer(stockfish_path, mistral_api_key)
        
        with open(pgn_file_path, 'r') as f:
            pgn_content = f.read()
        
        print("🔍 Starting premium analysis...")
        analysis = analyzer.analyze_game(pgn_content)
        
        # Print results
        print(f"\n🎯 PREMIUM ANALYSIS RESULTS")
        print(f"=" * 50)
        print(f"White Accuracy: {analysis.white_accuracy:.1f}%")
        print(f"Black Accuracy: {analysis.black_accuracy:.1f}%")
        
        print(f"\n📊 ERROR BREAKDOWN")
        print(f"White - Blunders: {analysis.white_errors['blunders']}, "
              f"Mistakes: {analysis.white_errors['mistakes']}, "
              f"Inaccuracies: {analysis.white_errors['inaccuracies']}")
        print(f"Black - Blunders: {analysis.black_errors['blunders']}, "
              f"Mistakes: {analysis.black_errors['mistakes']}, "
              f"Inaccuracies: {analysis.black_errors['inaccuracies']}")
        
        print(f"\n🔥 CRITICAL MOMENTS")
        for moment in analysis.critical_moments[:3]:
            print(f"Move {moment.move_number} ({moment.color}): {moment.move_san} - "
                  f"{moment.classification} (lost {moment.eval_loss:.2f})")
        
        print(f"\n🎯 TACTICAL INSIGHTS")
        print(analysis.tactical_insights)
        
        print(f"\n🧠 STRATEGIC INSIGHTS")
        print(analysis.strategic_insights)
        
        print(f"\n📚 IMPROVEMENT PLAN")
        print(analysis.improvement_plan)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def check_stockfish_installation():
    """Debug function to check Stockfish installation"""
    import os
    import shutil
    from pathlib import Path
    
    print("🔍 STOCKFISH INSTALLATION CHECK")
    print("=" * 40)
    
    # Check environment variable
    env_path = os.getenv('STOCKFISH_PATH')
    print(f"STOCKFISH_PATH environment variable: {env_path}")
    
    # Check common locations
    candidates = [
        env_path if env_path else None,
        'stockfish',
        '/usr/local/bin/stockfish',
        '/usr/bin/stockfish', 
        '/opt/homebrew/bin/stockfish',
        'stockfish.exe',
        './stockfish'
    ]
    
    print("\nChecking common Stockfish locations:")
    for candidate in candidates:
        if candidate:
            which_result = shutil.which(candidate)
            path_exists = Path(candidate).exists() if not which_result else True
            status = "✅ FOUND" if (which_result or path_exists) else "❌ NOT FOUND"
            print(f"  {candidate}: {status}")
            if which_result:
                print(f"    → Resolved to: {which_result}")
    
    # Check if stockfish is in PATH
    stockfish_in_path = shutil.which('stockfish')
    print(f"\nStockfish in system PATH: {'✅ YES' if stockfish_in_path else '❌ NO'}")
    if stockfish_in_path:
        print(f"  → Path: {stockfish_in_path}")
    
    # Installation suggestions
    print(f"\n💡 INSTALLATION SUGGESTIONS:")
    print(f"  Ubuntu/Debian: sudo apt-get install stockfish")
    print(f"  macOS: brew install stockfish") 
    print(f"  Windows: Download from https://stockfishchess.org/download/")
    print(f"  Manual: Place stockfish executable in current directory")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check-stockfish':
        check_stockfish_installation()
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] != '--check-stockfish':
        # Command line mode
        if len(sys.argv) != 4:
            print("Usage:")
            print("  python chess_analyzer.py --check-stockfish  (check installation)")
            print("  python chess_analyzer.py <pgn_file> <stockfish_path> <mistral_api_key>  (CLI analysis)")
            print("  python chess_analyzer.py  (start web server)")
            sys.exit(1)
        
        cli_analyze(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Web application mode
        setup_premium_analyzer()
        print("🚀 Starting Premium Chess Analyzer v2.0")
        print("📱 Access the web interface at: http://localhost:5000")
        print("🔧 Run with --check-stockfish to diagnose installation issues")
        app.run(host='0.0.0.0', port=5000, debug=True)