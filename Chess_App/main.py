
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv

from services.chess_analyzer import ChessAnalyzer
from services.ai_coach import AIChessCoach
from services.puzzle_generator import PuzzleGenerator
from services.skill_assessor import SkillAssessor
from services.opening_coach import OpeningCoach
from models.chess_models import *

load_dotenv()

app = FastAPI(title="Chess AI Coach Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
chess_analyzer = ChessAnalyzer()
ai_coach = AIChessCoach()
puzzle_generator = PuzzleGenerator()
skill_assessor = SkillAssessor()
opening_coach = OpeningCoach()

@app.on_event("startup")
async def startup_event():
    """Initialize all AI services on startup"""
    await chess_analyzer.initialize()
    await ai_coach.initialize()
    await skill_assessor.load_model()
    print("Chess AI Backend initialized successfully!")

# Chess Analysis Endpoints
@app.post("/analyze/position")
async def analyze_position(request: PositionAnalysisRequest):
    """Analyze a chess position with Stockfish and provide detailed evaluation"""
    try:
        analysis = await chess_analyzer.analyze_position(
            fen=request.fen,
            depth=request.depth,
            time_limit=request.time_limit
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/game")
async def analyze_game(request: GameAnalysisRequest):
    """Analyze entire game and provide move-by-move evaluation"""
    try:
        analysis = await chess_analyzer.analyze_game(
            pgn=request.pgn,
            player_color=request.player_color
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Coach Endpoints
@app.post("/coach/ask")
async def ask_coach(request: CoachQuestionRequest):
    """Ask the AI coach a question about a position or general chess concept"""
    try:
        response = await ai_coach.answer_question(
            question=request.question,
            context=request.context,
            fen=request.fen,
            player_level=request.player_level
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coach/explain-move")
async def explain_move(request: MoveExplanationRequest):
    """Get detailed explanation of why a move is good or bad"""
    try:
        explanation = await ai_coach.explain_move(
            fen=request.fen,
            move=request.move,
            context=request.context
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coach/suggest-improvement")
async def suggest_improvement(request: ImprovementRequest):
    """Analyze player's games and suggest specific areas for improvement"""
    try:
        suggestions = await ai_coach.suggest_improvements(
            recent_games=request.recent_games,
            current_rating=request.current_rating,
            weak_areas=request.weak_areas
        )
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Puzzle Generation Endpoints
@app.get("/puzzles/generate/{theme}")
async def generate_puzzles(theme: str, count: int = 10, rating_range: str = "1200-1800"):
    """Generate chess puzzles based on theme and difficulty"""
    try:
        min_rating, max_rating = map(int, rating_range.split('-'))
        puzzles = await puzzle_generator.generate_puzzles(
            theme=theme,
            count=count,
            min_rating=min_rating,
            max_rating=max_rating
        )
        return puzzles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/puzzles/check-solution")
async def check_puzzle_solution(request: PuzzleSolutionRequest):
    """Check if the player's solution to a puzzle is correct"""
    try:
        result = await puzzle_generator.check_solution(
            puzzle_id=request.puzzle_id,
            player_moves=request.player_moves
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Skill Assessment Endpoints
@app.post("/assess/tactical-skill")
async def assess_tactical_skill(request: TacticalAssessmentRequest):
    """Assess player's tactical skill based on puzzle performance"""
    try:
        assessment = await skill_assessor.assess_tactical_skill(
            puzzle_results=request.puzzle_results,
            time_spent=request.time_spent
        )
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess/game-strength")
async def assess_game_strength(request: GameStrengthRequest):
    """Assess overall game strength based on recent games"""
    try:
        assessment = await skill_assessor.assess_game_strength(
            games=request.games,
            current_rating=request.current_rating
        )
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Opening Coach Endpoints
@app.post("/opening/analyze")
async def analyze_opening(request: OpeningAnalysisRequest):
    """Analyze opening moves and provide suggestions"""
    try:
        analysis = await opening_coach.analyze_opening(
            moves=request.moves,
            color=request.color
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/opening/repertoire/{color}")
async def get_opening_repertoire(color: str, style: str = "balanced"):
    """Get personalized opening repertoire recommendations"""
    try:
        repertoire = await opening_coach.get_repertoire_suggestions(
            color=color,
            playing_style=style
        )
        return repertoire
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)