# models/game.py
import chess
import chess.pgn
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

class ChessGame:
    def __init__(self, game_id: str, white_player: str):
        self.game_id = game_id
        self.white_player = white_player
        self.black_player = None
        self.board = chess.Board()
        self.status = 'waiting'  # waiting, active, completed
        self.result = None  # white_wins, black_wins, draw
        self.moves_history = []
        self.created_at = datetime.now()
        self.last_move_time = datetime.now()
    
    def add_black_player(self, black_player: str):
        """Add black player to the game"""
        self.black_player = black_player
        self.status = 'active'
    
    def make_move(self, move_data: Dict, user_id: str) -> Dict[str, Any]:
        """
        Make a move in the game
        move_data should contain 'from' and 'to' squares, e.g., {'from': 'e2', 'to': 'e4'}
        """
        try:
            # Validate it's the player's turn
            if not self._is_players_turn(user_id):
                return {
                    'success': False,
                    'error': 'Not your turn'
                }
            
            # Convert move data to chess.Move
            from_square = chess.parse_square(move_data['from'])
            to_square = chess.parse_square(move_data['to'])
            
            # Handle promotion
            promotion = None
            if 'promotion' in move_data:
                promotion_piece = move_data['promotion'].lower()
                if promotion_piece == 'q':
                    promotion = chess.QUEEN
                elif promotion_piece == 'r':
                    promotion = chess.ROOK
                elif promotion_piece == 'b':
                    promotion = chess.BISHOP
                elif promotion_piece == 'n':
                    promotion = chess.KNIGHT
            
            move = chess.Move(from_square, to_square, promotion)
            
            # Validate move is legal
            if move not in self.board.legal_moves:
                return {
                    'success': False,
                    'error': 'Illegal move'
                }
            
            # Make the move
            self.board.push(move)
            self.moves_history.append({
                'move': move_data,
                'san': self.board.san(move),
                'fen': self.board.fen(),
                'timestamp': datetime.now().isoformat(),
                'player': user_id
            })
            
            self.last_move_time = datetime.now()
            
            # Check game status
            game_status = self._check_game_status()
            
            return {
                'success': True,
                'fen': self.board.fen(),
                'status': game_status['status'],
                'result': game_status.get('result'),
                'in_check': self.board.is_check(),
                'legal_moves': [str(move) for move in self.board.legal_moves],
                'last_move': move_data
            }
            
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid move format: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error making move: {str(e)}'
            }
    
    def _is_players_turn(self, user_id: str) -> bool:
        """Check if it's the given player's turn"""
        if self.status != 'active':
            return False
        
        # White's turn
        if self.board.turn == chess.WHITE:
            return user_id == self.white_player
        # Black's turn
        else:
            return user_id == self.black_player
    
    def _check_game_status(self) -> Dict[str, Any]:
        """Check current game status and result"""
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                # White is in checkmate, black wins
                self.result = 'black_wins'
            else:
                # Black is in checkmate, white wins
                self.result = 'white_wins'
            self.status = 'completed'
            return {'status': 'completed', 'result': self.result}
        
        elif self.board.is_stalemate():
            self.result = 'draw'
            self.status = 'completed'
            return {'status': 'completed', 'result': 'draw'}
        
        elif self.board.is_insufficient_material():
            self.result = 'draw'
            self.status = 'completed'
            return {'status': 'completed', 'result': 'draw'}
        
        elif self.board.is_seventyfive_moves():
            self.result = 'draw'
            self.status = 'completed'
            return {'status': 'completed', 'result': 'draw'}
        
        elif self.board.is_fivefold_repetition():
            self.result = 'draw'
            self.status = 'completed'
            return {'status': 'completed', 'result': 'draw'}
        
        else:
            return {'status': 'active'}
    
    def get_legal_moves(self) -> List[str]:
        """Get all legal moves in current position"""
        return [str(move) for move in self.board.legal_moves]
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get complete game state"""
        return {
            'game_id': self.game_id,
            'white_player': self.white_player,
            'black_player': self.black_player,
            'fen': self.board.fen(),
            'status': self.status,
            'result': self.result,
            'moves_history': self.moves_history,
            'legal_moves': self.get_legal_moves(),
            'in_check': self.board.is_check(),
            'turn': 'white' if self.board.turn == chess.WHITE else 'black',
            'created_at': self.created_at.isoformat(),
            'last_move_time': self.last_move_time.isoformat()
        }
    
    def resign(self, user_id: str) -> Dict[str, Any]:
        """Handle player resignation"""
        if user_id == self.white_player:
            self.result = 'black_wins'
        elif user_id == self.black_player:
            self.result = 'white_wins'
        else:
            return {'success': False, 'error': 'User not in this game'}
        
        self.status = 'completed'
        return {
            'success': True,
            'result': self.result,
            'reason': 'resignation'
        }
    
    def offer_draw(self, user_id: str) -> Dict[str, Any]:
        """Handle draw offer"""
        # Simple implementation - you might want to add draw offer state
        return {'success': True, 'message': 'Draw offered'}
    
    def accept_draw(self) -> Dict[str, Any]:
        """Accept a draw offer"""
        self.result = 'draw'
        self.status = 'completed'
        return {
            'success': True,
            'result': 'draw',
            'reason': 'agreement'
        }
    
    def to_pgn(self) -> str:
        """Convert game to PGN format"""
        game = chess.pgn.Game()
        
        # Set headers
        game.headers["Event"] = "Online Game"
        game.headers["Site"] = "Chess App"
        game.headers["Date"] = self.created_at.strftime("%Y.%m.%d")
        game.headers["White"] = self.white_player
        game.headers["Black"] = self.black_player or "Unknown"
        
        if self.result == 'white_wins':
            game.headers["Result"] = "1-0"
        elif self.result == 'black_wins':
            game.headers["Result"] = "0-1"
        elif self.result == 'draw':
            game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"
        
        # Add moves
        node = game
        temp_board = chess.Board()
        
        for move_data in self.moves_history:
            try:
                # Reconstruct the move from the stored data
                from_square = chess.parse_square(move_data['move']['from'])
                to_square = chess.parse_square(move_data['move']['to'])
                promotion = None
                
                if 'promotion' in move_data['move']:
                    promotion_piece = move_data['move']['promotion'].lower()
                    if promotion_piece == 'q':
                        promotion = chess.QUEEN
                    elif promotion_piece == 'r':
                        promotion = chess.ROOK
                    elif promotion_piece == 'b':
                        promotion = chess.BISHOP
                    elif promotion_piece == 'n':
                        promotion = chess.KNIGHT
                
                move = chess.Move(from_square, to_square, promotion)
                
                if move in temp_board.legal_moves:
                    temp_board.push(move)
                    node = node.add_variation(move)
            except:
                continue
        
        return str(game)