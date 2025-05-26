import chess
import chess.engine
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics

@dataclass
class SkillMetrics:
    rating: int
    accuracy: float
    tactical_strength: int
    positional_understanding: int
    endgame_skill: int
    time_management: float
    consistency: float

class SkillAssessor:
    def __init__(self):
        self.rating_ranges = {
            'beginner': (0, 1200),
            'intermediate': (1200, 1800),
            'advanced': (1800, 2200),
            'expert': (2200, 2600),
            'master': (2600, 3000)
        }
    
    def assess_game(self, pgn: str, time_control: Optional[str] = None) -> SkillMetrics:
        """Assess player skill from a single game"""
        game = chess.pgn.read_game(chess.io.StringIO(pgn))
        moves = list(game.mainline_moves())
        
        # Analyze key metrics
        accuracy = self._calculate_accuracy(moves)
        tactical_score = self._assess_tactical_play(moves)
        positional_score = self._assess_positional_play(moves)
        endgame_score = self._assess_endgame_play(moves)
        
        # Estimate rating based on combined metrics
        rating = self._estimate_rating(accuracy, tactical_score, positional_score)
        
        return SkillMetrics(
            rating=rating,
            accuracy=accuracy,
            tactical_strength=tactical_score,
            positional_understanding=positional_score,
            endgame_skill=endgame_score,
            time_management=self._assess_time_management(game, time_control),
            consistency=0.8  # Placeholder - would need multiple games
        )
    
    def assess_multiple_games(self, pgns: List[str]) -> SkillMetrics:
        """Assess skill from multiple games for better accuracy"""
        if not pgns:
            return SkillMetrics(1200, 0.5, 50, 50, 50, 1.0, 0.5)
        
        assessments = [self.assess_game(pgn) for pgn in pgns]
        
        return SkillMetrics(
            rating=int(statistics.mean(a.rating for a in assessments)),
            accuracy=statistics.mean(a.accuracy for a in assessments),
            tactical_strength=int(statistics.mean(a.tactical_strength for a in assessments)),
            positional_understanding=int(statistics.mean(a.positional_understanding for a in assessments)),
            endgame_skill=int(statistics.mean(a.endgame_skill for a in assessments)),
            time_management=statistics.mean(a.time_management for a in assessments),
            consistency=self._calculate_consistency([a.rating for a in assessments])
        )
    
    def _calculate_accuracy(self, moves: List[chess.Move]) -> float:
        """Calculate move accuracy using centipawn loss analysis"""
        if len(moves) < 10:
            return 0.7
        
        board = chess.Board()
        total_loss = 0
        move_count = 0
        
        for move in moves[:min(len(moves), 30)]:  # Analyze first 30 moves
            if board.is_legal(move):
                # Simple heuristic: evaluate common patterns
                if self._is_good_move(board, move):
                    total_loss += 20  # Good move
                elif self._is_blunder(board, move):
                    total_loss += 200  # Blunder
                else:
                    total_loss += 50  # Average move
                
                board.push(move)
                move_count += 1
        
        if move_count == 0:
            return 0.5
        
        avg_loss = total_loss / move_count
        accuracy = max(0.3, min(0.95, 1.0 - (avg_loss - 30) / 200))
        return round(accuracy, 3)
    
    def _is_good_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Simple heuristic for good moves"""
        # Check if move captures material
        if board.is_capture(move):
            return True
        
        # Check if move gives check
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            return True
        
        # Check if move develops a piece
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP] and move.from_square in [
            chess.B1, chess.G1, chess.C1, chess.F1, chess.B8, chess.G8, chess.C8, chess.F8
        ]:
            return True
        
        return False
    
    def _is_blunder(self, board: chess.Board, move: chess.Move) -> bool:
        """Simple heuristic for blunders"""
        # Check if move hangs material
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if the moved piece can be captured
        to_square = move.to_square
        if board_copy.is_attacked_by(not board.turn, to_square):
            piece = board_copy.piece_at(to_square)
            if piece and piece.piece_type > chess.PAWN:
                return True
        
        return False
    
    def _assess_tactical_play(self, moves: List[chess.Move]) -> int:
        """Assess tactical pattern recognition"""
        score = 50  # Base score
        board = chess.Board()
        
        for move in moves:
            if board.is_legal(move):
                if board.is_capture(move):
                    score += 2
                
                board_copy = board.copy()
                board_copy.push(move)
                if board_copy.is_check():
                    score += 3
                
                board.push(move)
        
        return min(100, max(0, score))
    
    def _assess_positional_play(self, moves: List[chess.Move]) -> int:
        """Assess positional understanding"""
        score = 50  # Base score
        board = chess.Board()
        
        development_count = 0
        castling_done = False
        
        for i, move in enumerate(moves):
            if board.is_legal(move):
                # Reward early development
                if i < 10:
                    piece = board.piece_at(move.from_square)
                    if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        development_count += 1
                
                # Check for castling
                if board.is_castling(move):
                    castling_done = True
                    score += 5
                
                board.push(move)
        
        score += development_count * 2
        if castling_done:
            score += 10
        
        return min(100, max(0, score))
    
    def _assess_endgame_play(self, moves: List[chess.Move]) -> int:
        """Assess endgame technique"""
        if len(moves) < 40:
            return 60  # Not enough moves to assess endgame
        
        score = 50
        board = chess.Board()
        
        # Play through the game to reach endgame
        for move in moves:
            if board.is_legal(move):
                board.push(move)
        
        # Simple endgame assessment based on material left
        piece_count = len(board.piece_map())
        if piece_count <= 8:  # Likely endgame
            if board.is_checkmate():
                score += 20  # Successfully converted endgame
            elif board.is_stalemate():
                score -= 10  # Missed win or drew losing position
        
        return min(100, max(0, score))
    
    def _estimate_rating(self, accuracy: float, tactical: int, positional: int) -> int:
        """Estimate ELO rating from metrics"""
        base_rating = 1000
        
        # Accuracy contribution (0-800 points)
        accuracy_points = int(accuracy * 800)
        
        # Tactical contribution (0-400 points)
        tactical_points = int((tactical / 100) * 400)
        
        # Positional contribution (0-400 points)
        positional_points = int((positional / 100) * 400)
        
        estimated_rating = base_rating + accuracy_points + tactical_points + positional_points
        
        return min(2800, max(600, estimated_rating))
    
    def _assess_time_management(self, game, time_control: Optional[str]) -> float:
        """Assess time management (simplified)"""
        # Would need actual time data from PGN
        return 1.0  # Placeholder
    
    def _calculate_consistency(self, ratings: List[int]) -> float:
        """Calculate rating consistency"""
        if len(ratings) < 2:
            return 0.5
        
        std_dev = statistics.stdev(ratings)
        # Lower standard deviation = higher consistency
        consistency = max(0.0, min(1.0, 1.0 - (std_dev / 400)))
        return round(consistency, 3)
    
    def get_skill_level(self, rating: int) -> str:
        """Get skill level category"""
        for level, (min_rating, max_rating) in self.rating_ranges.items():
            if min_rating <= rating < max_rating:
                return level
        return 'master'
    
    def get_improvement_areas(self, metrics: SkillMetrics) -> List[str]:
        """Suggest areas for improvement"""
        areas = []
        
        if metrics.tactical_strength < 60:
            areas.append("Tactical pattern recognition")
        if metrics.positional_understanding < 60:
            areas.append("Positional understanding")
        if metrics.endgame_skill < 60:
            areas.append("Endgame technique")
        if metrics.accuracy < 0.7:
            areas.append("Move accuracy and calculation")
        if metrics.consistency < 0.7:
            areas.append("Consistent play")
        
        return areas or ["Continue practicing all aspects of the game"]