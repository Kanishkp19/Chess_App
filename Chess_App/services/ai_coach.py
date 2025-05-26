# services/ai_coach.py
import requests
import chess
import chess.pgn
from typing import Dict, List, Optional
import json
import os

class AIChessCoach:
    def __init__(self):
        # Using Hugging Face's free inference API with a capable model
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        
        # Fallback to local model if API fails
        self.fallback_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        
    def _make_api_call(self, prompt: str, max_tokens: int = 300) -> str:
        """Make API call to Hugging Face with fallback"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
            else:
                # Try fallback model
                response = requests.post(self.fallback_url, headers=self.headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    return result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"
        
        return "Unable to generate analysis at this time."

    def analyze_move(self, fen: str, player_move: str, stockfish_analysis: Dict, 
                    player_rating: int = 1500) -> Dict:
        """Comprehensive move analysis with grandmaster-level feedback"""
        
        board = chess.Board(fen)
        try:
            move_obj = chess.Move.from_uci(player_move)
            move_san = board.san(move_obj)
        except:
            return {"error": "Invalid move format"}
        
        # Get move quality from Stockfish ranking
        best_moves = stockfish_analysis.get("best_moves", [])
        move_rank = self._get_move_rank(player_move, best_moves)
        eval_change = self._calculate_eval_change(stockfish_analysis)
        
        # Create focused prompt for grandmaster analysis
        prompt = self._create_grandmaster_prompt(fen, move_san, move_rank, eval_change, player_rating, best_moves)
        
        analysis = self._make_api_call(prompt, 400)
        
        return {
            "feedback": analysis,
            "move_quality": self._classify_move_quality(move_rank, eval_change),
            "move_rank": move_rank,
            "eval_change": eval_change,
            "position_phase": self._get_game_phase(board),
            "key_concepts": self._extract_concepts(board, move_obj, stockfish_analysis),
            "improvement_tip": self._get_improvement_tip(move_rank, eval_change, player_rating)
        }

    def analyze_position(self, fen: str, stockfish_data: Dict) -> Dict:
        """Deep position analysis like a grandmaster would provide"""
        
        board = chess.Board(fen)
        eval_score = stockfish_data.get("evaluation", {}).get("value", 0)
        best_moves = stockfish_data.get("best_moves", [])[:3]
        
        prompt = f"""As a chess grandmaster, analyze this position:

FEN: {fen}
Evaluation: {eval_score/100:.2f} pawns
Top moves: {', '.join([m.get('move', '') for m in best_moves])}

Provide analysis covering:
1. Strategic themes and plans for both sides
2. Key weaknesses and strengths
3. Critical squares and pieces
4. Tactical motifs present
5. Recommended approach for the player to move

Keep analysis concise but insightful."""

        analysis = self._make_api_call(prompt, 500)
        
        return {
            "analysis": analysis,
            "position_type": self._classify_position(board),
            "complexity": self._assess_complexity(stockfish_data),
            "tactical_motifs": self._find_tactical_motifs(board),
            "strategic_themes": self._identify_strategic_themes(board)
        }

    def answer_chess_question(self, question: str, context_fen: str = None, 
                            player_level: str = "intermediate") -> str:
        """Answer chess questions with educational approach"""
        
        context = f"\nPosition context: {context_fen}" if context_fen else ""
        
        prompt = f"""As a chess grandmaster and teacher, answer this question for a {player_level} player:

Question: {question}{context}

Provide a clear, educational answer with:
- Direct answer to the question
- Practical examples when relevant
- Specific advice for improvement
- Common mistakes to avoid

Keep the explanation appropriate for {player_level} level."""

        return self._make_api_call(prompt, 400)

    def create_study_plan(self, recent_games_analysis: List[Dict], 
                         current_rating: int, weak_areas: List[str]) -> Dict:
        """Generate personalized study plan based on game analysis"""
        
        common_mistakes = []
        for game in recent_games_analysis:
            common_mistakes.extend(game.get("mistakes", []))
        
        mistake_summary = ", ".join(set(common_mistakes[:5]))
        
        prompt = f"""Create a study plan for a {current_rating}-rated player:

Weak areas identified: {', '.join(weak_areas)}
Common mistakes: {mistake_summary}
Games analyzed: {len(recent_games_analysis)}

Provide:
1. Top 3 priority improvement areas
2. Specific study methods for each area
3. Recommended time allocation
4. Practice exercises
5. Timeline for expected improvement

Make recommendations practical and achievable."""

        study_plan = self._make_api_call(prompt, 500)
        
        return {
            "study_plan": study_plan,
            "priority_areas": self._prioritize_weaknesses(weak_areas, current_rating),
            "estimated_timeline": self._estimate_improvement_time(current_rating, weak_areas),
            "daily_routine": self._suggest_daily_routine(current_rating)
        }

    # Helper methods for efficient analysis
    def _create_grandmaster_prompt(self, fen: str, move: str, rank: int, 
                                  eval_change: float, rating: int, best_moves: List) -> str:
        """Create focused prompt for move analysis"""
        
        quality = "excellent" if rank == 1 else "good" if rank <= 3 else "questionable" if rank <= 5 else "poor"
        alternative = best_moves[0].get('move', 'N/A') if best_moves else 'N/A'
        
        return f"""As a chess grandmaster, analyze this move:

Position: {fen}
Move played: {move}
Move quality: {quality} (rank #{rank if rank else '6+'})
Evaluation change: {eval_change:+.2f} pawns
Player rating: {rating}
Best alternative: {alternative}

Provide grandmaster-level feedback:
- What this move accomplishes or fails to do
- Why it's good/bad compared to alternatives  
- Key strategic/tactical concepts involved
- Specific advice for this rating level

Be encouraging but honest about mistakes."""

    def _get_move_rank(self, move: str, best_moves: List[Dict]) -> int:
        """Get the ranking of the played move"""
        for i, best_move in enumerate(best_moves, 1):
            if best_move.get("move") == move:
                return i
        return 6  # Not in top 5

    def _calculate_eval_change(self, analysis: Dict) -> float:
        """Calculate evaluation change from the move"""
        eval_before = analysis.get("evaluation_before", {})
        eval_after = analysis.get("evaluation_after", {})
        
        if eval_before.get("type") == "mate" or eval_after.get("type") == "mate":
            return 0.0
        
        before = eval_before.get("value", 0) / 100.0
        after = eval_after.get("value", 0) / 100.0
        return after - before

    def _classify_move_quality(self, rank: int, eval_change: float) -> str:
        """Classify move quality"""
        if rank == 1:
            return "excellent"
        elif rank <= 2:
            return "very_good"
        elif rank <= 3:
            return "good"
        elif eval_change > -0.3:
            return "acceptable"
        elif eval_change > -1.0:
            return "questionable"
        else:
            return "poor"

    def _get_game_phase(self, board: chess.Board) -> str:
        """Determine game phase"""
        moves = len(board.move_stack)
        pieces = len(board.piece_map())
        
        if moves < 15:
            return "opening"
        elif pieces <= 12:
            return "endgame"
        else:
            return "middlegame"

    def _extract_concepts(self, board: chess.Board, move: chess.Move, analysis: Dict) -> List[str]:
        """Extract key chess concepts from the move"""
        concepts = []
        
        if board.is_capture(move):
            concepts.append("capture")
        if board.gives_check(move):
            concepts.append("check")
        if self._is_development_move(board, move):
            concepts.append("development")
        if self._is_tactical_move(board, move):
            concepts.append("tactics")
        
        return concepts

    def _get_improvement_tip(self, rank: int, eval_change: float, rating: int) -> str:
        """Generate specific improvement tip"""
        if rank > 5:
            return "Calculate all candidate moves before deciding"
        elif eval_change < -1.0:
            return "Look for all opponent threats before moving"
        elif rating < 1200:
            return "Focus on basic tactical patterns"
        elif rating < 1600:
            return "Study typical middlegame plans"
        else:
            return "Analyze grandmaster games in similar positions"

    def _classify_position(self, board: chess.Board) -> str:
        """Classify position type"""
        if board.is_check():
            return "tactical"
        elif len(board.piece_map()) <= 10:
            return "endgame"
        elif self._has_opposite_castling(board):
            return "attacking"
        else:
            return "positional"

    def _assess_complexity(self, stockfish_data: Dict) -> str:
        """Assess position complexity"""
        best_moves = stockfish_data.get("best_moves", [])
        if len(best_moves) < 2:
            return "simple"
        
        eval_diff = abs(best_moves[0].get("evaluation", {}).get("value", 0) - 
                       best_moves[1].get("evaluation", {}).get("value", 0))
        
        return "complex" if eval_diff < 30 else "moderate" if eval_diff < 100 else "simple"

    def _find_tactical_motifs(self, board: chess.Board) -> List[str]:
        """Identify tactical motifs in position"""
        motifs = []
        
        if board.is_check():
            motifs.append("check")
        
        # Look for pins, forks, skewers (simplified detection)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if self._creates_pin(board, square):
                    motifs.append("pin")
                if self._creates_fork(board, square):
                    motifs.append("fork")
        
        return motifs

    def _identify_strategic_themes(self, board: chess.Board) -> List[str]:
        """Identify strategic themes"""
        themes = []
        
        if self._has_pawn_majority(board):
            themes.append("pawn_majority")
        if self._has_weak_squares(board):
            themes.append("weak_squares")
        if self._has_piece_activity_imbalance(board):
            themes.append("piece_activity")
        
        return themes

    def _prioritize_weaknesses(self, weak_areas: List[str], rating: int) -> List[str]:
        """Prioritize improvement areas by rating"""
        priority_map = {
            1000: ["tactics", "basic_endgames", "opening_principles"],
            1300: ["tactics", "calculation", "positional_play"],
            1600: ["strategy", "endgames", "opening_theory"],
            1900: ["deep_calculation", "advanced_endgames", "preparation"]
        }
        
        for threshold, priorities in sorted(priority_map.items()):
            if rating <= threshold:
                return [area for area in priorities if area in weak_areas][:3]
        
        return weak_areas[:3]

    def _estimate_improvement_time(self, rating: int, weak_areas: List[str]) -> str:
        """Estimate time for improvement"""
        base_time = 4 if rating < 1400 else 6 if rating < 1800 else 8
        return f"{base_time}-{base_time+2} weeks per area with consistent study"

    def _suggest_daily_routine(self, rating: int) -> Dict:
        """Suggest daily practice routine"""
        if rating < 1400:
            return {
                "tactics": "15-20 minutes",
                "endgames": "10 minutes", 
                "games": "1-2 games with analysis"
            }
        else:
            return {
                "tactics": "20-30 minutes",
                "endgames": "15 minutes",
                "strategy": "15 minutes",
                "games": "2-3 games with deep analysis"
            }

    # Simplified helper methods
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type == chess.PAWN:
            return False
        
        back_ranks = {chess.WHITE: [0, 1], chess.BLACK: [6, 7]}
        from_rank = move.from_square // 8
        
        return (from_rank in back_ranks[piece.color] and 
                piece.piece_type in [chess.KNIGHT, chess.BISHOP])

    def _is_tactical_move(self, board: chess.Board, move: chess.Move) -> bool:
        return (board.is_capture(move) or 
                board.gives_check(move) or 
                board.is_castling(move))

    def _has_opposite_castling(self, board: chess.Board) -> bool:
        # Simplified check for opposite-side castling
        return False  # Placeholder

    def _creates_pin(self, board: chess.Board, square: int) -> bool:
        # Simplified pin detection
        return False  # Placeholder

    def _creates_fork(self, board: chess.Board, square: int) -> bool:
        # Simplified fork detection  
        return False  # Placeholder

    def _has_pawn_majority(self, board: chess.Board) -> bool:
        # Check for pawn majorities on different sides
        return False  # Placeholder

    def _has_weak_squares(self, board: chess.Board) -> bool:
        # Detect weak squares in position
        return False  # Placeholder

    def _has_piece_activity_imbalance(self, board: chess.Board) -> bool:
        # Check for piece activity differences
        return False  # Placeholder