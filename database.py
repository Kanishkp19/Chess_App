import chess
from datetime import datetime
from typing import Dict, Optional, Any
import json
import os
import shutil
from pathlib import Path

class GameDatabase:
    def __init__(self, data_file: str = "games_data.json"):
        self.data_file = data_file
        self.backup_file = f"{data_file}.backup"
        self.games = {}
        self.load_games()
    
    def load_games(self):
        """Load games from JSON file"""
        # Try to load from main file first
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._reconstruct_games_from_data(data)
                print(f"Loaded {len(self.games)} games from {self.data_file}")
                return
            except Exception as e:
                print(f"Error loading main file: {e}")
                # Try backup file
                if os.path.exists(self.backup_file):
                    try:
                        with open(self.backup_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self._reconstruct_games_from_data(data)
                        print(f"Loaded {len(self.games)} games from backup file")
                        return
                    except Exception as backup_error:
                        print(f"Error loading backup file: {backup_error}")
        
        print("No valid game data found, starting with empty database")
        self.games = {}
    
    def _reconstruct_games_from_data(self, data: dict):
        """Reconstruct game objects from saved data"""
        self.games = {}
        for game_id, game_data in data.items():
            try:
                # Reconstruct board from FEN
                board = chess.Board(game_data['board_fen'])
                
                # Replay moves to ensure board state is correct
                temp_board = chess.Board()
                for move_uci in game_data['move_history']:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in temp_board.legal_moves:
                            temp_board.push(move)
                    except:
                        print(f"Warning: Invalid move {move_uci} in game {game_id}")
                        break
                
                # Use the replayed board if it matches the saved FEN
                if temp_board.fen() == game_data['board_fen']:
                    board = temp_board
                
                self.games[game_id] = {
                    'board': board,
                    'player_color': game_data['player_color'],
                    'difficulty': game_data['difficulty'],
                    'move_history': game_data['move_history'],
                    'created_at': game_data.get('created_at', datetime.now().isoformat()),
                    'last_updated': game_data.get('last_updated', datetime.now().isoformat())
                }
            except Exception as e:
                print(f"Error reconstructing game {game_id}: {e}")
                continue
    
    def save_games(self):
        """Save games to JSON file with backup"""
        try:
            # Prepare data for saving
            data = {}
            for game_id, game in self.games.items():
                data[game_id] = {
                    'board_fen': game['board'].fen(),
                    'player_color': game['player_color'],
                    'difficulty': game['difficulty'],
                    'move_history': game['move_history'],
                    'created_at': game['created_at'],
                    'last_updated': datetime.now().isoformat()
                }
            
            # Create backup of existing file
            if os.path.exists(self.data_file):
                try:
                    shutil.copy2(self.data_file, self.backup_file)
                except Exception as e:
                    print(f"Warning: Could not create backup: {e}")
            
            # Write to temporary file first
            temp_file = f"{self.data_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomically replace the main file
            if os.name == 'nt':  # Windows
                if os.path.exists(self.data_file):
                    os.remove(self.data_file)
                os.rename(temp_file, self.data_file)
            else:  # Unix-like systems
                os.rename(temp_file, self.data_file)
            
            print(f"Successfully saved {len(self.games)} games to {self.data_file}")
            
        except Exception as e:
            print(f"Error saving games: {e}")
            # Clean up temp file if it exists
            temp_file = f"{self.data_file}.tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise e
    
    def force_save(self):
        """Force save all games immediately"""
        print("Force saving all games...")
        self.save_games()
        print("Force save completed")
    
    def create_game(self, player_color: str, difficulty: int) -> str:
        """Create a new game and return game ID"""
        game_id = f"game_{len(self.games)}_{int(datetime.now().timestamp())}"
        board = chess.Board()
        
        self.games[game_id] = {
            'board': board,
            'player_color': player_color,
            'difficulty': difficulty,
            'move_history': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Auto-save after creating game
        self.save_games()
        print(f"Created new game: {game_id}")
        return game_id
    
    def get_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get game by ID"""
        return self.games.get(game_id)
    
    def update_game(self, game_id: str):
        """Update game's last_updated timestamp and save"""
        if game_id in self.games:
            self.games[game_id]['last_updated'] = datetime.now().isoformat()
            self.save_games()
            print(f"Updated game: {game_id}")
    
    def delete_game(self, game_id: str) -> bool:
        """Delete a game"""
        if game_id in self.games:
            del self.games[game_id]
            self.save_games()
            print(f"Deleted game: {game_id}")
            return True
        return False
    
    def get_all_games_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info about all games for debugging"""
        game_info = {}
        for game_id, game in self.games.items():
            game_info[game_id] = {
                'player_color': game['player_color'],
                'difficulty': game['difficulty'],
                'moves_played': len(game['move_history']),
                'game_over': game['board'].is_game_over(),
                'current_fen': game['board'].fen(),
                'created_at': game['created_at'],
                'last_updated': game['last_updated']
            }
        return game_info
    
    def cleanup_old_games(self, max_age_hours: int = 24):
        """Clean up games older than specified hours"""
        current_time = datetime.now()
        games_to_delete = []
        
        for game_id, game in self.games.items():
            try:
                last_updated = datetime.fromisoformat(game['last_updated'])
                age_hours = (current_time - last_updated).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    games_to_delete.append(game_id)
            except:
                # If timestamp parsing fails, mark for deletion
                games_to_delete.append(game_id)
        
        for game_id in games_to_delete:
            del self.games[game_id]
        
        if games_to_delete:
            self.save_games()
            print(f"Cleaned up {len(games_to_delete)} old games")
        
        return len(games_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        total_games = len(self.games)
        active_games = sum(1 for game in self.games.values() if not game['board'].is_game_over())
        completed_games = total_games - active_games
        
        difficulty_counts = {}
        color_counts = {'white': 0, 'black': 0}
        
        for game in self.games.values():
            diff = game['difficulty']
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            color_counts[game['player_color']] += 1
        
        return {
            'total_games': total_games,
            'active_games': active_games,
            'completed_games': completed_games,
            'difficulty_distribution': difficulty_counts,
            'color_distribution': color_counts,
            'data_file': self.data_file,
            'file_exists': os.path.exists(self.data_file),
            'file_size': os.path.getsize(self.data_file) if os.path.exists(self.data_file) else 0
        }
    
    def export_games(self, filename: str = None):
        """Export all games to a specific file"""
        if filename is None:
            filename = f"chess_games_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Prepare comprehensive data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_games': len(self.games),
                'games': {}
            }
            
            for game_id, game in self.games.items():
                export_data['games'][game_id] = {
                    'board_fen': game['board'].fen(),
                    'player_color': game['player_color'],
                    'difficulty': game['difficulty'],
                    'move_history': game['move_history'],
                    'game_over': game['board'].is_game_over(),
                    'result': self._get_game_result(game['board']),
                    'created_at': game['created_at'],
                    'last_updated': game['last_updated']
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"Exported {len(self.games)} games to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting games: {e}")
            return None
    
    def _get_game_result(self, board: chess.Board) -> str:
        """Get human-readable game result"""
        if not board.is_game_over():
            return "In progress"
        elif board.is_checkmate():
            return "Checkmate - " + ("White wins" if board.turn == chess.BLACK else "Black wins")
        elif board.is_stalemate():
            return "Stalemate"
        elif board.is_insufficient_material():
            return "Draw - Insufficient material"
        elif board.is_seventyfive_moves():
            return "Draw - 75 move rule"
        elif board.is_fivefold_repetition():
            return "Draw - Repetition"
        else:
            return "Draw"
    
    def verify_data_integrity(self):
        """Verify that all games can be properly reconstructed"""
        print("Verifying data integrity...")
        issues = []
        
        for game_id, game in self.games.items():
            try:
                # Check if board is valid
                if not isinstance(game['board'], chess.Board):
                    issues.append(f"Game {game_id}: Invalid board object")
                    continue
                
                # Check if move history matches board state
                temp_board = chess.Board()
                for i, move_uci in enumerate(game['move_history']):
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move not in temp_board.legal_moves:
                            issues.append(f"Game {game_id}: Illegal move at position {i}: {move_uci}")
                            break
                        temp_board.push(move)
                    except Exception as e:
                        issues.append(f"Game {game_id}: Invalid move format at position {i}: {move_uci}")
                        break
                
                # Check if final position matches
                if temp_board.fen() != game['board'].fen():
                    issues.append(f"Game {game_id}: Board state doesn't match move history")
                
            except Exception as e:
                issues.append(f"Game {game_id}: General error - {e}")
        
        if issues:
            print(f"Found {len(issues)} data integrity issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Data integrity check passed - all games are valid")
        
        return issues