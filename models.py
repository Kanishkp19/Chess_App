"""Pydantic models for request/response validation"""

from pydantic import BaseModel
from typing import Optional


class MoveRequest(BaseModel):
    game_id: str
    move: str


class NewGameRequest(BaseModel):
    player_color: str = "white"
    difficulty: int = 1