import chess
import chess.pgn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class OpeningRecommendation:
    name: str
    moves: str
    reason: str
    difficulty: str
    success_rate: float

@dataclass
class OpeningAnalysis:
    opening_name: str
    accuracy: float
    common_mistakes: List[str]
    suggested_improvements: List[str]
    repertoire_gaps: List[str]

class OpeningCoach:
    def __init__(self):
        self.opening_database = self._load_opening_database()
        self.opening_principles = self._load_opening_principles()
    
    def analyze_opening_play(self, pgns: List[str]) -> OpeningAnalysis:
        """Analyze player's opening repertoire and performance"""
        opening_stats = defaultdict(list)
        all_mistakes = []
        
        for pgn in pgns:
            game = chess.pgn.read_game(chess.io.StringIO(pgn))
            if not game:
                continue
                
            opening = self._identify_opening(game)
            moves = list(game.mainline_moves())[:15]  # First 15 moves
            
            mistakes = self._find_opening_mistakes(moves)
            accuracy = self._calculate_opening_accuracy(moves)
            
            opening_stats[opening].append(accuracy)
            all_mistakes.extend(mistakes)
        
        # Get most played opening
        main_opening = max(opening_stats.keys(), key=lambda x: len(opening_stats[x])) if opening_stats else "Unknown"
        avg_accuracy = sum(opening_stats[main_opening]) / len(opening_stats[main_opening]) if opening_stats[main_opening] else 0.5
        
        return OpeningAnalysis(
            opening_name=main_opening,
            accuracy=avg_accuracy,
            common_mistakes=list(set(all_mistakes))[:5],
            suggested_improvements=self._get_improvement_suggestions(main_opening, avg_accuracy),
            repertoire_gaps=self._identify_repertoire_gaps(opening_stats)
        )
    
    def recommend_openings(self, skill_level: str, color: str, style: str = "balanced") -> List[OpeningRecommendation]:
        """Recommend openings based on player profile"""
        recommendations = []
        
        if color.lower() == "white":
            recommendations.extend(self._get_white_recommendations(skill_level, style))
        else:
            recommendations.extend(self._get_black_recommendations(skill_level, style))
        
        return recommendations[:3]  # Top 3 recommendations
    
    def get_opening_training_positions(self, opening_name: str) -> List[Dict]:
        """Generate training positions for specific opening"""
        positions = []
        
        if opening_name in self.opening_database:
            main_line = self.opening_database[opening_name]["moves"]
            board = chess.Board()
            
            # Play main line and create training positions
            moves = main_line.split()
            for i in range(0, min(len(moves), 12), 2):
                if i < len(moves):
                    move = chess.Move.from_uci(moves[i])
                    if board.is_legal(move):
                        board.push(move)
                        
                        # Create training position
                        positions.append({
                            "fen": board.fen(),
                            "move_number": (i // 2) + 1,
                            "key_ideas": self._get_position_ideas(board, opening_name),
                            "common_alternatives": self._get_alternative_moves(board, moves[i+1:])
                        })
        
        return positions[:6]  # Return 6 key positions
    
    def _load_opening_database(self) -> Dict:
        """Load opening database with main lines"""
        return {
            "Italian Game": {
                "moves": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
                "ideas": ["Control center", "Develop pieces", "Castle early"],
                "difficulty": "beginner"
            },
            "Spanish Opening": {
                "moves": "e2e4 e7e5 g1f3 b8c6 f1b5",
                "ideas": ["Pressure e5 pawn", "Maintain central tension"],
                "difficulty": "intermediate"
            },
            "Queen's Gambit": {
                "moves": "d2d4 d7d5 c2c4",
                "ideas": ["Control center with pawns", "Develop queenside"],
                "difficulty": "intermediate"
            },
            "King's Indian Defense": {
                "moves": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
                "ideas": ["Fianchetto king's bishop", "Counter-attack"],
                "difficulty": "advanced"
            },
            "French Defense": {
                "moves": "e2e4 e7e6",
                "ideas": ["Solid pawn structure", "Counter-play on queenside"],
                "difficulty": "intermediate"
            },
            "Sicilian Defense": {
                "moves": "e2e4 c7c5",
                "ideas": ["Asymmetrical play", "Counter-attack on queenside"],
                "difficulty": "advanced"
            },
            "Caro-Kann Defense": {
                "moves": "e2e4 c7c6",
                "ideas": ["Solid structure", "Safe development"],
                "difficulty": "intermediate"
            },
            "English Opening": {
                "moves": "c2c4",
                "ideas": ["Flexible development", "Control d5 square"],
                "difficulty": "advanced"
            }
        }
    
    def _load_opening_principles(self) -> Dict:
        """Load fundamental opening principles"""
        return {
            "development": ["Develop knights before bishops", "Castle early", "Don't move same piece twice"],
            "center": ["Control central squares", "Occupy center with pawns", "Challenge opponent's center"],
            "safety": ["Castle within first 10 moves", "Don't weaken king position", "Connect rooks"],
            "time": ["Don't waste moves", "Develop with purpose", "Don't chase opponent's pieces"]
        }
    
    def _identify_opening(self, game) -> str:
        """Identify opening from game moves"""
        moves = []
        node = game
        move_count = 0
        
        while node.variations and move_count < 6:
            move = str(node.variations[0].move)
            moves.append(move)
            node = node.variations[0]
            move_count += 1
        
        moves_str = " ".join(moves)
        
        # Simple pattern matching
        for opening, data in self.opening_database.items():
            opening_moves = data["moves"].replace("e2e4", "e4").replace("e7e5", "e5")  # Simplified notation
            if opening_moves[:20] in moves_str[:20]:
                return opening
        
        # Basic classification
        if "e4" in moves_str[:5]:
            if "e5" in moves_str[:10]:
                return "King's Pawn Game"
            elif "c5" in moves_str[:10]:
                return "Sicilian Defense"
            else:
                return "King's Pawn Opening"
        elif "d4" in moves_str[:5]:
            if "d5" in moves_str[:10]:
                return "Queen's Pawn Game"
            else:
                return "Queen's Pawn Opening"
        elif "Nf3" in moves_str[:5] or "c4" in moves_str[:5]:
            return "English Opening"
        
        return "Other Opening"
    
    def _find_opening_mistakes(self, moves: List[chess.Move]) -> List[str]:
        """Identify common opening mistakes"""
        mistakes = []
        board = chess.Board()
        move_count = 0
        developed_pieces = set()
        castled = False
        
        for move in moves:
            if not board.is_legal(move):
                continue
                
            move_count += 1
            
            # Check for early queen moves
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.QUEEN and move_count <= 4:
                mistakes.append("Developing queen too early")
            
            # Check for moving same piece twice
            if move.from_square in developed_pieces and move_count <= 8:
                mistakes.append("Moving same piece multiple times in opening")
            
            if piece:
                developed_pieces.add(move.to_square)
            
            # Check castling timing
            if board.is_castling(move):
                castled = True
            elif move_count > 12 and not castled:
                mistakes.append("Delaying castling too long")
            
            # Check for weakening moves
            if piece and piece.piece_type == chess.PAWN:
                if move.to_square in [chess.H3, chess.G3, chess.F3, chess.H6, chess.G6, chess.F6]:
                    mistakes.append("Weakening king position with pawn moves")
            
            board.push(move)
        
        return mistakes
    
    def _calculate_opening_accuracy(self, moves: List[chess.Move]) -> float:
        """Calculate opening accuracy based on principles"""
        if not moves:
            return 0.5
        
        score = 0.7  # Base score
        board = chess.Board()
        developed_pieces = 0
        castled = False
        center_control = 0
        
        for i, move in enumerate(moves[:10]):
            if not board.is_legal(move):
                continue
            
            piece = board.piece_at(move.from_square)
            
            # Reward development
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                developed_pieces += 1
                score += 0.02
            
            # Reward castling
            if board.is_castling(move):
                castled = True
                score += 0.05
            
            # Reward center control
            if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                center_control += 1
                score += 0.03
            
            board.push(move)
        
        # Penalties
        if not castled and len(moves) > 8:
            score -= 0.1
        if developed_pieces < 2:
            score -= 0.1
        
        return min(0.95, max(0.3, score))
    
    def _get_improvement_suggestions(self, opening: str, accuracy: float) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if accuracy < 0.6:
            suggestions.append("Focus on basic opening principles: development, center control, and king safety")
        
        if opening in self.opening_database:
            suggestions.append(f"Study the main line of {opening} more deeply")
            suggestions.append(f"Practice typical middlegame positions arising from {opening}")
        
        suggestions.append("Learn 2-3 solid opening systems rather than many superficial ones")
        suggestions.append("Study opening traps and tactical motifs in your openings")
        
        return suggestions[:3]
    
    def _identify_repertoire_gaps(self, opening_stats: Dict) -> List[str]:
        """Identify gaps in opening repertoire"""
        gaps = []
        
        # Check for missing responses to common openings
        played_openings = set(opening_stats.keys())
        
        if "French Defense" not in played_openings and "Caro-Kann Defense" not in played_openings:
            gaps.append("Need response to 1.e4 besides e5")
        
        if "King's Indian Defense" not in played_openings and "Queen's Gambit" not in played_openings:
            gaps.append("Need solid response to 1.d4")
        
        if len(played_openings) < 3:
            gaps.append("Expand opening repertoire for more variety")
        
        return gaps[:3]
    
    def _get_white_recommendations(self, skill_level: str, style: str) -> List[OpeningRecommendation]:
        """Get opening recommendations for White"""
        recommendations = []
        
        if skill_level == "beginner":
            recommendations.append(OpeningRecommendation(
                name="Italian Game",
                moves="1.e4 e5 2.Nf3 Nc6 3.Bc4",
                reason="Simple development, clear plans, many tactical opportunities",
                difficulty="Easy",
                success_rate=0.75
            ))
            recommendations.append(OpeningRecommendation(
                name="Queen's Gambit",
                moves="1.d4 d5 2.c4",
                reason="Solid positional play, teaches pawn structure concepts",
                difficulty="Easy-Medium",
                success_rate=0.72
            ))
        
        elif skill_level == "intermediate":
            if style == "aggressive":
                recommendations.append(OpeningRecommendation(
                    name="Spanish Opening",
                    moves="1.e4 e5 2.Nf3 Nc6 3.Bb5",
                    reason="Rich in strategic and tactical ideas, maintains pressure",
                    difficulty="Medium",
                    success_rate=0.78
                ))
            else:
                recommendations.append(OpeningRecommendation(
                    name="English Opening",
                    moves="1.c4",
                    reason="Flexible system, leads to various pawn structures",
                    difficulty="Medium",
                    success_rate=0.73
                ))
        
        else:  # advanced
            recommendations.append(OpeningRecommendation(
                name="Nimzo-Larsen Attack",
                moves="1.b3",
                reason="Hypermodern approach, surprise value, flexible setup",
                difficulty="Hard",
                success_rate=0.69
            ))
        
        return recommendations
    
    def _get_black_recommendations(self, skill_level: str, style: str) -> List[OpeningRecommendation]:
        """Get opening recommendations for Black"""
        recommendations = []
        
        if skill_level == "beginner":
            recommendations.append(OpeningRecommendation(
                name="French Defense",
                moves="1.e4 e6",
                reason="Solid structure, clear plans, teaches pawn chains",
                difficulty="Easy-Medium",
                success_rate=0.71
            ))
            recommendations.append(OpeningRecommendation(
                name="Queen's Gambit Declined",
                moves="1.d4 d5 2.c4 e6",
                reason="Classical setup, solid and reliable",
                difficulty="Easy-Medium",
                success_rate=0.74
            ))
        
        elif skill_level == "intermediate":
            if style == "aggressive":
                recommendations.append(OpeningRecommendation(
                    name="Sicilian Defense",
                    moves="1.e4 c5",
                    reason="Counter-attacking chances, unbalanced positions",
                    difficulty="Medium-Hard",
                    success_rate=0.69
                ))
            else:
                recommendations.append(OpeningRecommendation(
                    name="Caro-Kann Defense",
                    moves="1.e4 c6",
                    reason="Solid and safe, good pawn structure",
                    difficulty="Medium",
                    success_rate=0.73
                ))
        
        else:  # advanced
            recommendations.append(OpeningRecommendation(
                name="King's Indian Defense",
                moves="1.d4 Nf6 2.c4 g6",
                reason="Dynamic counterplay, complex middlegame positions",
                difficulty="Hard",
                success_rate=0.67
            ))
        
        return recommendations
    
    def _get_position_ideas(self, board: chess.Board, opening: str) -> List[str]:
        """Get key ideas for current position"""
        ideas = []
        
        if opening in self.opening_database:
            ideas.extend(self.opening_database[opening].get("ideas", []))
        
        # Add general ideas based on position
        if not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK):
            ideas.append("Both sides have castled - focus on piece activity")
        elif board.turn == chess.WHITE and board.has_castling_rights(chess.WHITE):
            ideas.append("Consider castling for king safety")
        
        return ideas[:3]
    
    def _get_alternative_moves(self, board: chess.Board, remaining_moves: List[str]) -> List[str]:
        """Get alternative moves for training"""
        alternatives = []
        
        # Generate some legal moves as alternatives
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        
        for move in legal_moves[:3]:
            alternatives.append(str(move))
        
        return alternatives