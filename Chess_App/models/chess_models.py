# models/chess_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class DifficultyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"

class PlayerColor(str, Enum):
    white = "white"
    black = "black"

class GameStatus(str, Enum):
    ongoing = "ongoing"
    checkmate = "checkmate"
    stalemate = "stalemate"
    insufficient_material = "insufficient_material"
    fifty_moves = "fifty_moves"
    threefold_repetition = "threefold_repetition"

# Existing models (from your original code)
class PositionAnalysisRequest(BaseModel):
    fen: str
    depth: int = 15
    time_limit: float = 1.0

class GameAnalysisRequest(BaseModel):
    pgn: str
    player_color: str

class CoachQuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    fen: Optional[str] = None
    player_level: str = "intermediate"

class MoveExplanationRequest(BaseModel):
    fen: str
    move: str
    context: Optional[str] = None

class ImprovementRequest(BaseModel):
    recent_games: List[str]
    current_rating: int
    weak_areas: List[str]

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
    color: str

# New enhanced models for Stockfish integration
class MoveAnalysisRequest(BaseModel):
    fen: str
    move: str  # Move in UCI format (e.g., "e2e4")
    depth: int = 15
    multiPV: int = 3  # Number of best moves to analyze

class GameMoveRequest(BaseModel):
    fen: str
    difficulty: DifficultyLevel = DifficultyLevel.intermediate
    time_limit: float = 1.0
    opening_book: bool = True

class TutorAnalysisRequest(BaseModel):
    fen: str
    player_move: str
    player_level: str = "intermediate"
    explain_alternatives: bool = True

class GameSessionRequest(BaseModel):
    session_id: str
    player_color: PlayerColor
    difficulty: DifficultyLevel = DifficultyLevel.intermediate
    tutoring_enabled: bool = True

class GameMoveSubmission(BaseModel):
    session_id: str
    move: str  # Player's move in UCI format
    fen_before: str

# Response models
class EvaluationScore(BaseModel):
    type: str  # "cp" for centipawns, "mate" for mate
    value: int

class BestMove(BaseModel):
    move: Optional[str]
    evaluation: EvaluationScore
    pv: List[str]  # Principal variation
    depth: int = 0

class MoveAnalysisResponse(BaseModel):
    move: str
    evaluation_before: EvaluationScore
    evaluation_after: EvaluationScore
    best_moves_before: List[BestMove]
    best_moves_after: List[BestMove]
    move_classification: Dict[str, bool]

class PositionFeatures(BaseModel):
    material_balance: int
    king_safety: Dict[str, int]
    pawn_structure: Dict[str, Any]
    piece_activity: Dict[str, int]
    tactical_motifs: List[str]

class PositionAnalysisResponse(BaseModel):
    evaluation: EvaluationScore
    best_moves: List[BestMove]
    position_features: PositionFeatures
    coach_insights: Dict[str, Any]

class ComputerMoveResponse(BaseModel):
    move: str
    evaluation: EvaluationScore
    difficulty: str
    thinking_time: float

class GameSessionResponse(BaseModel):
    session_created: bool
    starting_fen: str
    computer_first_move: Optional[str] = None
    fen_after_computer_move: Optional[str] = None

class GameStatusInfo(BaseModel):
    status: GameStatus
    winner: Optional[str] = None
    in_check: bool = False

class MoveQualityFeedback(BaseModel):
    move_quality: str
    feedback: str
    key_concepts: List[str]
    improvement_tips: List[str]

class MoveTutorResponse(BaseModel):
    move_quality: str
    feedback: MoveQualityFeedback
    alternatives: List[BestMove]
    stockfish_analysis: MoveAnalysisResponse

class GameMoveResponse(BaseModel):
    player_move_accepted: bool
    fen_after_player_move: str
    game_status: GameStatusInfo
    move_analysis: Optional[MoveTutorResponse] = None
    computer_move: Optional[str] = None
    fen_after_computer_move: Optional[str] = None
    final_game_status: Optional[GameStatusInfo] = None

class CoachResponse(BaseModel):
    answer: str
    player_level: str
    related_concepts: List[str]

class MoveExplanationResponse(BaseModel):
    explanation: str
    move: str
    concepts: List[str]

class ImprovementSuggestion(BaseModel):
    improvement_plan: str
    priority_areas: List[str]
    study_time_estimate: str

class PositionInsights(BaseModel):
    analysis: str
    position_type: str
    complexity: str
    learning_focus: List[str]

# Puzzle models
class PuzzleTheme(str, Enum):
    mate_in_1 = "mate_in_1"
    mate_in_2 = "mate_in_2"
    mate_in_3 = "mate_in_3"
    tactics = "tactics"
    endgame = "endgame"
    opening = "opening"
    middlegame = "middlegame"
    pin = "pin"
    fork = "fork"
    skewer = "skewer"
    discovery = "discovery"
    deflection = "deflection"
    decoy = "decoy"
    clearance = "clearance"
    sacrifice = "sacrifice"

class ChessPuzzle(BaseModel):
    id: str
    fen: str
    moves: List[str]  # Solution moves
    theme: PuzzleTheme
    rating: int
    description: str

class PuzzleGenerationResponse(BaseModel):
    puzzles: List[ChessPuzzle]
    count: int
    theme: PuzzleTheme
    rating_range: str

class PuzzleSolutionResponse(BaseModel):
    correct: bool
    solution: List[str]
    explanation: str
    player_moves: List[str]
    feedback: str

# Skill assessment models
class PuzzleResult(BaseModel):
    puzzle_id: str
    solved: bool
    time_taken: float
    attempts: int
    rating: int

class TacticalSkillAssessment(BaseModel):
    overall_rating: int
    strengths: List[str]
    weaknesses: List[str]
    recommended_themes: List[PuzzleTheme]
    accuracy_percentage: float
    average_solve_time: float

class GameStrengthAssessment(BaseModel):
    estimated_rating: int
    rating_confidence: float
    playing_strengths: List[str]
    areas_for_improvement: List[str]
    game_phase_analysis: Dict[str, Dict[str, Any]]

# Opening coach models
class OpeningMove(BaseModel):
    move: str
    frequency: float
    win_percentage: float
    draw_percentage: float
    loss_percentage: float

class OpeningAnalysis(BaseModel):
    opening_name: str
    eco_code: str
    evaluation: EvaluationScore
    main_line: List[str]
    alternatives: List[OpeningMove]
    typical_plans: List[str]
    key_squares: List[str]

class OpeningRepertoire(BaseModel):
    color: PlayerColor
    style: str
    recommended_openings: List[OpeningAnalysis]
    study_order: List[str]
    time_investment: Dict[str, str]

# Error handling models
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Utility models for complex responses
class LearningModule(BaseModel):
    title: str
    description: str
    difficulty: DifficultyLevel
    estimated_time: str
    prerequisites: List[str]
    objectives: List[str]

class StudyPlan(BaseModel):
    player_rating: int
    weak_areas: List[str]
    modules: List[LearningModule]
    total_time_estimate: str
    priority_order: List[str]

class PerformanceMetrics(BaseModel):
    rating_change: int
    games_played: int
    win_percentage: float
    tactical_accuracy: float
    opening_knowledge: float
    endgame_skill: float
    time_management: float

# Session management models
class ActiveGameSession(BaseModel):
    session_id: str
    player_color: PlayerColor
    difficulty: DifficultyLevel
    tutoring_enabled: bool
    current_fen: str
    move_count: int
    start_time: str
    last_activity: str
    game_status: GameStatus