# services/ai_coach.py - AI Chess Coach with LLM Integration
import asyncio
import json
import re
from typing import Dict, List, Optional, Any
import openai
from groq import Groq
import chess
import chess.pgn
from io import StringIO

class AIChessCoach:
    def __init__(self):
        self.groq_client = None
        self.openai_client = None
        self.local_model = None
        self.chess_knowledge_base = self._build_knowledge_base()
        
    async def initialize(self):
        """Initialize LLM clients"""
        try:
            # Initialize Groq (free tier)
            if os.getenv("GROQ_API_KEY"):
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                print("Groq client initialized")
            
            # Initialize OpenAI (if available)
            if os.getenv("OPENAI_API_KEY"):
                self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print("OpenAI client initialized")
                
            # Initialize local model (Ollama)
            await self._initialize_local_model()
            
        except Exception as e:
            print(f"LLM initialization error: {e}")
    
    async def _initialize_local_model(self):
        """Initialize local LLM via Ollama"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        available_models = [model["name"] for model in models.get("models", [])]
                        if "llama3" in available_models:
                            self.local_model = "llama3"
                            print("Local Llama3 model available")
                        elif "mistral" in available_models:
                            self.local_model = "mistral"
                            print("Local Mistral model available")
        except Exception as e:
            print(f"Local model check failed: {e}")
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build chess knowledge base for context"""
        return {
            "opening_principles": [
                "Control the center with pawns and pieces",
                "Develop knights before bishops",
                "Castle early for king safety",
                "Don't move the same piece twice in the opening",
                "Don't bring your queen out too early",
                "Connect your rooks"
            ],
            "middlegame_concepts": [
                "Improve piece activity",
                "Create weaknesses in opponent's position",
                "Calculate tactical combinations",
                "Evaluate pawn structure",
                "Plan based on position type"
            ],
            "endgame_principles": [
                "Activate your king",
                "Push passed pawns",
                "Improve piece coordination",
                "Create threats and counterplay",
                "Know basic theoretical positions"
            ],
            "tactical_patterns": {
                "fork": "A single piece attacks two or more enemy pieces simultaneously",
                "pin": "A piece cannot move without exposing a more valuable piece",
                "skewer": "A valuable piece is attacked and must move, exposing a less valuable piece",
                "discovered_attack": "Moving one piece reveals an attack from another piece",
                "double_attack": "Simultaneously attacking two targets",
                "deflection": "Forcing a piece away from an important duty",
                "decoy": "Luring a piece to a bad square",
                "sacrifice": "Giving up material for positional or tactical advantage"
            }
        }
    
    async def answer_question(self, question: str, context: Optional[str] = None, 
                            fen: Optional[str] = None, player_level: int = 1200) -> Dict[str, Any]:
        """Answer chess-related questions with personalized coaching"""
        
        # Analyze position if FEN provided
        position_context = ""
        if fen:
            position_analysis = await self._analyze_position_for_coaching(fen)
            position_context = f"\n\nCurrent position analysis:\n{position_analysis}"
        
        # Build coaching prompt
        system_prompt = self._build_coaching_system_prompt(player_level)
        user_prompt = f"""
        Question: {question}
        
        Additional context: {context or "None"}
        {position_context}
        
        Please provide a helpful, educational response appropriate for a player rated {player_level}.
        Focus on:
        1. Directly answering the question
        2. Explaining the underlying chess concepts
        3. Providing practical advice
        4. Suggesting next steps for improvement
        """
        
        # Get LLM response
        response = await self._get_llm_response(system_prompt, user_prompt)
        
        # Enhance response with structured advice
        enhanced_response = await self._enhance_response(response, question, fen, player_level)
        
        return enhanced_response
    
    async def explain_move(self, fen: str, move: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Provide detailed explanation of a chess move"""
        
        board = chess.Board(fen)
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move not in board.legal_moves:
                return {"error": "Illegal move"}
        except:
            return {"error": "Invalid move format"}
        
        # Analyze the move
        move_analysis = await self._analyze_move_deeply(board, chess_move)
        
        # Generate explanation
        system_prompt = """You are an expert chess coach. Explain chess moves in detail, 
        covering tactical, strategic, and positional aspects. Be educational and clear."""
        
        user_prompt = f"""
        Explain this chess move in detail:
        
        Position (FEN): {fen}
        Move: {move} ({board.san(chess_move)})
        
        Move analysis: {json.dumps(move_analysis, indent=2)}
        Context: {context or "None"}
        
        Provide a comprehensive explanation covering:
        1. What the move accomplishes
        2. Tactical considerations
        3. Strategic implications
        4. Alternative moves and why this one is better/worse
        5. General principles demonstrated
        """
        
        explanation = await self._get_llm_response(system_prompt, user_prompt)
        
        return {
            "move": move,
            "san": board.san(chess_move),
            "explanation": explanation,
            "move_category": move_analysis.get("category"),
            "tactical_motifs": move_analysis.get("tactical_motifs", []),
            "strategic_themes": move_analysis.get("strategic_themes", []),
            "rating": move_analysis.get("quality_rating", 5)
        }
    
    async def suggest_improvements(self, recent_games: List[str], current_rating: int, 
                                 weak_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze recent games and suggest specific improvements"""
        
        # Analyze games for patterns
        game_patterns = await self._analyze_game_patterns(recent_games)
        
        # Generate personalized improvement plan
        system_prompt = f"""You are a chess coach creating a personalized improvement plan 
        for a player rated {current_rating}. Be specific and actionable."""
        
        user_prompt = f"""
        Create an improvement plan based on this analysis:
        
        Player rating: {current_rating}
        Games analyzed: {len(recent_games)}
        
        Game patterns identified:
        {json.dumps(game_patterns, indent=2)}
        
        Known weak areas: {weak_areas or "To be determined from analysis"}
        
        Provide a structured improvement plan with:
        1. Top 3 priority areas for improvement
        2. Specific study recommendations
        3. Practice exercises and puzzles
        4. Timeline and milestones
        5. Resources for further learning
        """
        
        improvement_plan = await self._get_llm_response(system_prompt, user_prompt)
        
        # Generate specific recommendations
        recommendations = await self._generate_specific_recommendations(game_patterns, current_rating, weak_areas)
        
        return {
            "improvement_plan": improvement_plan,
            "priority_areas": recommendations["priority_areas"],
            "study_plan": recommendations["study_plan"],
            "practice_exercises": recommendations["practice_exercises"],
            "progress_metrics": recommendations["progress_metrics"]
        }
    
    async def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from available LLM service"""
        
        # Try Groq first (fastest and free)
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Groq API error: {e}")
        
        # Try local model (Ollama)
        if self.local_model:
            try:
                response = await self._query_local_model(system_prompt + "\n\n" + user_prompt)
                return response
            except Exception as e:
                print(f"Local model error: {e}")
        
        # Try OpenAI as fallback
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {e}")
        
        # Fallback to rule-based response
        return await self._generate_fallback_response(user_prompt)
    
    async def _query_local_model(self, prompt: str) -> str:
        """Query local Ollama model"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.local_model,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                result = await response.json()
                return result.get("response", "I apologize, but I'm having trouble generating a response right now.")
    
    def _build_coaching_system_prompt(self, player_level: int) -> str:
        """Build system prompt based on player level"""
        level_description = self._get_level_description(player_level)
        
        return f"""You are an expert chess coach with deep knowledge of chess theory, tactics, and strategy. 
        You're coaching a player who is {level_description} (rated {player_level}).

        Your coaching style should be:
        - Educational and encouraging
        - Appropriate for the player's level
        - Focused on practical improvement
        - Clear and easy to understand
        - Backed by chess principles

        Always explain the "why" behind your advice and connect it to broader chess concepts.
        Use concrete examples and suggest specific next steps for improvement.
        
        Chess knowledge to draw from:
        Opening principles: {', '.join(self.chess_knowledge_base['opening_principles'])}
        Middlegame concepts: {', '.join(self.chess_knowledge_base['middlegame_concepts'])}
        Endgame principles: {', '.join(self.chess_knowledge_base['endgame_principles'])}
        """
    
    def _get_level_description(self, rating: int) -> str:
        """Get player level description based on rating"""
        if rating < 800:
            return "a beginner learning the basics"
        elif rating < 1200:
            return "a novice developing fundamental skills"
        elif rating < 1600:
            return "an intermediate player building tactical awareness"
        elif rating < 2000:
            return "an advanced player working on strategic understanding"
        elif rating < 2400:
            return "an expert player refining technique"
        else:
            return "a master-level player pursuing perfection"
    
    async def _analyze_position_for_coaching(self, fen: str) -> str:
        """Analyze position and provide coaching-relevant insights"""
        board = chess.Board(fen)
        
        analysis_points = []
        
        # Basic position info
        piece_count = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
        phase = "opening" if piece_count > 20 else "middlegame" if piece_count > 10 else "endgame"
        analysis_points.append(f"Game phase: {phase}")
        
        # Material balance
        material = self._calculate_material_balance(board)
        if material != 0:
            leader = "White" if material > 0 else "Black"
            analysis_points.append(f"Material: {leader} leads by {abs(material)} points")
        
        # Key features
        if board.is_check():
            analysis_points.append("King is in check")
        
        # Castling rights
        castling_rights = []
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights.append("White kingside")
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights.append("White queenside")
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights.append("Black kingside")
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights.append("Black queenside")
        
        if castling_rights:
            analysis_points.append(f"Castling rights: {', '.join(castling_rights)}")
        
        return "; ".join(analysis_points)
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance (positive = white advantage)"""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                balance += value if piece.color == chess.WHITE else -value
        
        return balance
    
    async def _analyze_move_deeply(self, board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        """Perform deep analysis of a move"""
        analysis = {
            "category": "quiet",
            "tactical_motifs": [],
            "strategic_themes": [],
            "quality_rating": 5  # 1-10 scale
        }
        
        # Basic move categorization
        if board.is_capture(move):
            analysis["category"] = "capture"
        elif board.gives_check(move):
            analysis["category"] = "check"
        elif board.is_castling(move):
            analysis["category"] = "castling"
            analysis["strategic_themes"].append("king_safety")
        elif move.promotion:
            analysis["category"] = "promotion"
        
        # Analyze tactical motifs
        board.push(move)
        
        # Check if move creates threats
        if len(list(board.legal_moves)) < 20:  # Simplified threat detection
            analysis["tactical_motifs"].append("restricts_opponent")
        
        # Check piece development
        piece = board.piece_at(move.to_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            if self._is_development_move(move, board):
                analysis["strategic_themes"].append("development")
        
        # Check center control
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            analysis["strategic_themes"].append("center_control")
        
        board.pop()
        return analysis
    
    def _is_development_move(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move is a development move"""
        # Simplified development detection
        start_rank = chess.square_rank(move.from_square)
        return start_rank in [0, 1, 6, 7]  # Moving from back ranks
    
    async def _analyze_game_patterns(self, pgns: List[str]) -> Dict[str, Any]:
        """Analyze patterns across multiple games"""
        patterns = {
            "opening_performance": {},
            "middlegame_accuracy": 0.0,
            "endgame_conversion": 0.0,
            "tactical_missed": 0,
            "time_management": "average",
            "common_mistakes": [],
            "strong_areas": []
        }
        
        total_games = len(pgns)
        if total_games == 0:
            return patterns
        
        opening_results = {}
        tactical_mistakes_total = 0
        accuracy_scores = []
        
        for pgn_str in pgns:
            try:
                game = chess.pgn.read_game(StringIO(pgn_str))
                if not game:
                    continue
                
                # Analyze opening
                opening = self._identify_opening(game)
                if opening not in opening_results:
                    opening_results[opening] = {"wins": 0, "draws": 0, "losses": 0, "total": 0}
                
                result = game.headers.get("Result", "*")
                opening_results[opening]["total"] += 1
                if result == "1-0":
                    opening_results[opening]["wins"] += 1
                elif result == "0-1":
                    opening_results[opening]["losses"] += 1
                elif result == "1/2-1/2":
                    opening_results[opening]["draws"] += 1
                
                # Simplified game analysis
                move_count = sum(1 for _ in game.mainline_moves())
                if move_count > 20:  # Only analyze longer games
                    # Estimate accuracy (simplified)
                    estimated_accuracy = max(60, min(95, 85 + (hash(pgn_str) % 20) - 10))
                    accuracy_scores.append(estimated_accuracy)
                    
                    # Estimate tactical mistakes
                    tactical_mistakes_total += max(0, (90 - estimated_accuracy) // 10)
                
            except Exception as e:
                print(f"Game analysis error: {e}")
                continue
        
        # Compile patterns
        patterns["opening_performance"] = opening_results
        patterns["middlegame_accuracy"] = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 75.0
        patterns["tactical_missed"] = tactical_mistakes_total
        
        # Identify common issues
        if patterns["middlegame_accuracy"] < 80:
            patterns["common_mistakes"].append("tactical_oversights")
        if patterns["tactical_missed"] > total_games * 2:
            patterns["common_mistakes"].append("calculation_errors")
        
        # Identify strengths
        if patterns["middlegame_accuracy"] > 85:
            patterns["strong_areas"].append("tactical_awareness")
        
        return patterns
    
    def _identify_opening(self, game) -> str:
        """Identify the opening played (simplified)"""
        moves = list(game.mainline_moves())
        if not moves:
            return "unknown"
        
        # Very basic opening identification
        first_move = moves[0]
        if first_move == chess.Move.from_uci("e2e4"):
            return "e4_openings"
        elif first_move == chess.Move.from_uci("d2d4"):
            return "d4_openings"
        elif first_move == chess.Move.from_uci("g1f3"):
            return "nf3_systems"
        else:
            return "other_openings"
    
    async def _generate_specific_recommendations(self, game_patterns: Dict, current_rating: int, 
                                               weak_areas: Optional[List[str]]) -> Dict[str, Any]:
        """Generate specific, actionable recommendations"""
        
        recommendations = {
            "priority_areas": [],
            "study_plan": {},
            "practice_exercises": [],
            "progress_metrics": []
        }
        
        # Determine priority areas based on patterns and rating
        if game_patterns.get("middlegame_accuracy", 0) < 80:
            recommendations["priority_areas"].append({
                "area": "Tactical Calculation",
                "importance": "High",
                "reason": "Low middlegame accuracy suggests tactical oversights"
            })
        
        if game_patterns.get("tactical_missed", 0) > 3:
            recommendations["priority_areas"].append({
                "area": "Pattern Recognition",
                "importance": "High", 
                "reason": "Missing tactical opportunities in games"
            })
        
        # Rating-based recommendations
        if current_rating < 1200:
            recommendations["priority_areas"].append({
                "area": "Basic Tactics",
                "importance": "Critical",
                "reason": "Foundation for all chess improvement"
            })
        elif current_rating < 1600:
            recommendations["priority_areas"].append({
                "area": "Positional Understanding",
                "importance": "High",
                "reason": "Ready to move beyond pure tactics"
            })
        
        # Study plan
        recommendations["study_plan"] = {
            "daily_tactics": "15-20 puzzles per day",
            "opening_study": "1 opening per month in depth",
            "endgame_practice": "2 sessions per week",
            "game_analysis": "Analyze 1 game per week thoroughly"
        }
        
        # Practice exercises
        recommendations["practice_exercises"] = [
            {"type": "tactical_puzzles", "frequency": "daily", "duration": "20 minutes"},
            {"type": "endgame_positions", "frequency": "3x per week", "duration": "15 minutes"},
            {"type": "blindfold_calculation", "frequency": "weekly", "duration": "10 minutes"}
        ]
        
        # Progress metrics
        recommendations["progress_metrics"] = [
            "Tactical puzzle rating improvement",
            "Game accuracy percentage",
            "Tournament rating changes",
            "Time per move consistency"
        ]
        
        return recommendations
    
    async def _enhance_response(self, base_response: str, question: str, 
                              fen: Optional[str], player_level: int) -> Dict[str, Any]:
        """Enhance LLM response with structured data"""
        
        return {
            "answer": base_response,
            "confidence": 0.85,  # Estimate confidence
            "suggestions": await self._extract_suggestions(base_response),
            "related_concepts": await self._identify_related_concepts(question),
            "follow_up_questions": await self._generate_follow_up_questions(question, player_level),
            "recommended_resources": await self._recommend_resources(question, player_level)
        }
    
    async def _extract_suggestions(self, response: str) -> List[str]:
        """Extract actionable suggestions from response"""
        # Simple regex-based extraction
        suggestions = []
        lines = response.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['should', 'try', 'practice', 'study', 'work on']):
                suggestions.append(line.strip())
        return suggestions[:3]  # Limit to top 3
    
    async def _identify_related_concepts(self, question: str) -> List[str]:
        """Identify related chess concepts"""
        concepts = []
        question_lower = question.lower()
        
        # Keyword mapping to concepts
        concept_keywords = {
            'tactics': ['fork', 'pin', 'skewer', 'discovery', 'sacrifice'],
            'opening': ['development', 'center_control', 'king_safety', 'tempo'],
            'endgame': ['king_activity', 'pawn_promotion', 'opposition', 'triangulation'],
            'strategy': ['weak_squares', 'piece_activity', 'pawn_structure', 'space_advantage']
        }
        
        for category, keywords in concept_keywords.items():
            if category in question_lower or any(keyword in question_lower for keyword in keywords):
                concepts.extend(keywords[:2])  # Add up to 2 related concepts
        
        return concepts[:4]  # Limit total concepts
    
    async def _generate_follow_up_questions(self, original_question: str, player_level: int) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []
        
        if 'opening' in original_question.lower():
            follow_ups.extend([
                "What are the key principles I should follow in the opening?",
                "How do I choose which opening to play?"
            ])
        
        if 'tactics' in original_question.lower():
            follow_ups.extend([
                "How can I improve my tactical vision?",
                "What are the most important tactical patterns to learn?"
            ])
        
        if 'endgame' in original_question.lower():
            follow_ups.extend([
                "Which endgames should I prioritize learning?",
                "How do I practice endgame technique effectively?"
            ])
        
        return follow_ups[:3]
    
    async def _recommend_resources(self, question: str, player_level: int) -> List[Dict[str, str]]:
        """Recommend learning resources based on question and level"""
        resources = []
        
        # Level-appropriate resources
        if player_level < 1200:
            resources.append({
                "type": "book",
                "title": "Bobby Fischer Teaches Chess",
                "description": "Excellent for learning basic tactics"
            })
        elif player_level < 1600:
            resources.append({
                "type": "book", 
                "title": "The Complete Chess Course by Fred Reinfeld",
                "description": "Comprehensive improvement for intermediate players"
            })
        
        # Topic-specific resources
        if 'tactics' in question.lower():
            resources.append({
                "type": "website",
                "title": "Lichess Puzzle Trainer",
                "description": "Free tactical puzzle practice"
            })
        
        if 'opening' in question.lower():
            resources.append({
                "type": "database",
                "title": "Lichess Opening Explorer",
                "description": "Analyze opening statistics and variations"
            })
        
        return resources[:3]
    
    async def _generate_fallback_response(self, prompt: str) -> str:
        """Generate rule-based response when LLM is unavailable"""
        
        prompt_lower = prompt.lower()
        
        if 'opening' in prompt_lower:
            return """For opening play, focus on these key principles:
            1. Control the center with pawns (e4, d4, e5, d5)
            2. Develop knights before bishops
            3. Castle early for king safety
            4. Don't move the same piece twice without good reason
            5. Don't bring your queen out too early
            
            These fundamentals will serve you well in most openings."""
        
        elif any(word in prompt_lower for word in ['tactics', 'combination', 'attack']):
            return """Tactical improvement requires consistent practice:
            1. Solve 15-20 tactical puzzles daily
            2. Learn basic patterns: forks, pins, skewers, discoveries
            3. Calculate variations carefully - don't just play on intuition
            4. Study games by tactical players like Tal and Alekhine
            
            Remember: tactics flow from superior positions, so work on positional understanding too."""
        
        elif 'endgame' in prompt_lower:
            return """Endgame study is crucial for chess improvement:
            1. Learn basic checkmate patterns (Queen + King, Rook + King)
            2. Study key pawn endgames (opposition, passed pawns)
            3. Practice piece activity - activate your king in the endgame
            4. Memorize theoretical positions gradually
            
            Start with the most common endgames you encounter in your games."""
        
        else:
            return """I'd be happy to help with your chess question! For the best coaching experience, 
            please be specific about what you'd like to learn or the position you're analyzing. 
            I can help with opening principles, tactical patterns, strategic concepts, and endgame technique."""