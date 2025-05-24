from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class Color(str, Enum):
    WHITE = "white"
    BLACK = "black"

class PuzzleTheme(str, Enum):
    FORK = "fork"
    PIN = "pin"
    SKEWER = "skewer"
    DISCOVERY = "discovery"
    DEFLECTION = "deflection"
    SACRIFICE = "sacrifice"
    MATE_IN_1 = "mate_in_1"
    MATE_IN_2 = "mate_in_2"
    MATE_IN_3 = "mate_in_3"
    ENDGAME = "endgame"

# Request Models
class PositionAnalysisRequest(BaseModel):
    fen: str = Field(..., description="FEN string of the position")
    depth: int = Field(default=18, ge=1, le=25)
    time_limit: Optional[float] = Field(default=2.0, description="Analysis time in seconds")

class GameAnalysisRequest(BaseModel):
    pgn: str = Field(..., description="PGN of the game")
    player_color: Optional[Color] = None

class CoachQuestionRequest(BaseModel):
    question: str = Field(..., description="Player's question")
    fen: Optional[str] = None
    context: Optional[str] = None
    player_level: int = Field(default=1200, description="Player's rating")

class MoveExplanationRequest(BaseModel):
    fen: str
    move: str
    context: Optional[str] = None

class ImprovementRequest(BaseModel):
    recent_games: List[str] = Field(..., description="List of recent game PGNs")
    current_rating: int
    weak_areas: Optional[List[str]] = None

class PuzzleSolutionRequest(BaseModel):
    puzzle_id: str
    player_moves: List[str]

class TacticalAssessmentRequest(BaseModel):
    puzzle_results: List[Dict[str, Any]]
    time_spent: List[float]

class GameStrengthRequest(BaseModel):
    games: List[str]
    current_rating: int

class OpeningAnalysisRequest(BaseModel):
    moves: List[str]
    color: Color

# Response Models
class PositionEvaluation(BaseModel):
    score: float
    mate_in: Optional[int] = None
    best_moves: List[Dict[str, Any]]
    position_type: str
    key_features: List[str]

class CoachResponse(BaseModel):
    answer: str
    suggestions: List[str]
    related_concepts: List[str]
    confidence: float

class PuzzleData(BaseModel):
    id: str
    fen: str
    solution: List[str]
    theme: str
    rating: int
    description: str


