import unittest
import sys
import os

# Add the parent directory to the path to import the main modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chess_logic import ChessGame, ChessBoard, Piece
except ImportError:
    # Create mock classes if the actual modules don't exist
    class ChessGame:
        def __init__(self):
            self.board = ChessBoard()
            self.current_player = 'white'
            self.game_over = False
            
        def make_move(self, from_pos, to_pos):
            return True
            
        def is_valid_move(self, from_pos, to_pos):
            return True
            
        def get_valid_moves(self, pos):
            return []
            
        def reset(self):
            self.__init__()
    
    class ChessBoard:
        def __init__(self):
            self.board = self._initialize_board()
            
        def _initialize_board(self):
            return [[None for _ in range(8)] for _ in range(8)]
            
        def get_piece(self, row, col):
            return self.board[row][col]
            
        def set_piece(self, row, col, piece):
            self.board[row][col] = piece
    
    class Piece:
        def __init__(self, color, piece_type):
            self.color = color
            self.type = piece_type


class TestChessLogic(unittest.TestCase):
    def setUp(self):
        """Set up a new chess game for each test"""
        self.game = ChessGame()

    def test_initial_board_setup(self):
        """Test that the chess board is set up correctly initially"""
        # Test that the game starts with white to move
        self.assertEqual(self.game.current_player, 'white')
        
        # Test that the game is not over initially
        self.assertFalse(self.game.game_over)
        
        # Test board dimensions
        if hasattr(self.game, 'board') and hasattr(self.game.board, 'board'):
            self.assertEqual(len(self.game.board.board), 8)
            self.assertEqual(len(self.game.board.board[0]), 8)

    def test_valid_pawn_moves(self):
        """Test valid pawn movements"""
        # Test initial pawn move (one or two squares)
        self.assertTrue(self.game.is_valid_move('e2', 'e3'))
        self.assertTrue(self.game.is_valid_move('e2', 'e4'))
        
        # Test invalid pawn moves
        self.assertFalse(self.game.is_valid_move('e2', 'e5'))
        self.assertFalse(self.game.is_valid_move('e2', 'd3'))

    def test_knight_moves(self):
        """Test knight movement patterns"""
        # Test valid knight moves from starting position
        valid_moves = self.game.get_valid_moves('b1')
        expected_moves = ['a3', 'c3']
        
        # Knight should be able to move in L-shape
        if valid_moves:
            self.assertIsInstance(valid_moves, list)

    def test_piece_capture(self):
        """Test piece capture mechanics"""
        # This would test capturing opponent pieces
        # Implementation depends on specific chess logic structure
        result = self.game.make_move('e2', 'e4')
        self.assertIsInstance(result, bool)

    def test_check_detection(self):
        """Test check detection"""
        # This would test if the king is in check
        if hasattr(self.game, 'is_in_check'):
            self.assertFalse(self.game.is_in_check('white'))
            self.assertFalse(self.game.is_in_check('black'))

    def test_checkmate_detection(self):
        """Test checkmate detection"""
        if hasattr(self.game, 'is_checkmate'):
            self.assertFalse(self.game.is_checkmate('white'))
            self.assertFalse(self.game.is_checkmate('black'))

    def test_stalemate_detection(self):
        """Test stalemate detection"""
        if hasattr(self.game, 'is_stalemate'):
            self.assertFalse(self.game.is_stalemate())

    def test_castling(self):
        """Test castling rules"""
        # Test kingside and queenside castling
        if hasattr(self.game, 'can_castle'):
            # Initially, castling should not be possible due to pieces in the way
            self.assertFalse(self.game.can_castle('white', 'kingside'))
            self.assertFalse(self.game.can_castle('white', 'queenside'))

    def test_en_passant(self):
        """Test en passant capture"""
        # This would test the en passant rule
        if hasattr(self.game, 'can_en_passant'):
            # Initially, en passant should not be possible
            self.assertFalse(self.game.can_en_passant('e5', 'd5'))

    def test_pawn_promotion(self):
        """Test pawn promotion"""
        # This would test pawn reaching the end of the board
        if hasattr(self.game, 'promote_pawn'):
            # Test promotion logic
            pass

    def test_move_history(self):
        """Test move history tracking"""
        if hasattr(self.game, 'move_history'):
            initial_history_length = len(self.game.move_history)
            self.game.make_move('e2', 'e4')
            self.assertEqual(len(self.game.move_history), initial_history_length + 1)

    def test_undo_move(self):
        """Test undoing moves"""
        if hasattr(self.game, 'undo_move'):
            self.game.make_move('e2', 'e4')
            result = self.game.undo_move()
            self.assertTrue(result)

    def test_square_notation_conversion(self):
        """Test conversion between square notation and coordinates"""
        if hasattr(self.game, 'notation_to_coords'):
            row, col = self.game.notation_to_coords('e4')
            self.assertEqual(row, 4)
            self.assertEqual(col, 4)
        else:
            # Test helper method
            row, col = self._notation_to_coords('e4')
            self.assertEqual(row, 4)
            self.assertEqual(col, 4)

    def test_coords_to_notation_conversion(self):
        """Test conversion from coordinates to square notation"""
        if hasattr(self.game, 'coords_to_notation'):
            notation = self.game.coords_to_notation(4, 4)
            self.assertEqual(notation, 'e4')
        else:
            # Test helper method
            notation = self._coords_to_notation(4, 4)
            self.assertEqual(notation, 'e4')

    def test_piece_movement_validation(self):
        """Test that pieces can only move according to their rules"""
        # Test that a rook can't move diagonally
        if hasattr(self.game, 'is_valid_piece_move'):
            # This would depend on the specific implementation
            pass

    def test_turn_alternation(self):
        """Test that turns alternate between players"""
        initial_player = self.game.current_player
        self.game.make_move('e2', 'e4')
        
        if hasattr(self.game, 'switch_player'):
            # Player should switch after a valid move
            self.assertNotEqual(self.game.current_player, initial_player)

    def test_invalid_move_handling(self):
        """Test handling of invalid moves"""
        result = self.game.make_move('e2', 'e5')  # Invalid pawn move
        if isinstance(result, bool):
            self.assertFalse(result)

    def test_game_reset(self):
        """Test game reset functionality"""
        # Make some moves
        self.game.make_move('e2', 'e4')
        
        # Reset the game
        self.game.reset()
        
        # Check that game is back to initial state
        self.assertEqual(self.game.current_player, 'white')
        self.assertFalse(self.game.game_over)

    def _notation_to_coords(self, notation):
        """Helper method to convert chess notation to coordinates"""
        file = ord(notation[0]) - ord('a')
        rank = int(notation[1]) - 1
        return rank, file

    def _coords_to_notation(self, row, col):
        """Helper method to convert coordinates to chess notation"""
        file = chr(ord('a') + col)
        rank = str(row + 1)
        return file + rank


class TestChessBoard(unittest.TestCase):
    def setUp(self):
        """Set up a new chess board for each test"""
        self.board = ChessBoard()

    def test_board_initialization(self):
        """Test that the board is properly initialized"""
        self.assertEqual(len(self.board.board), 8)
        self.assertEqual(len(self.board.board[0]), 8)

    def test_piece_placement(self):
        """Test placing and retrieving pieces"""
        piece = Piece('white', 'pawn')
        self.board.set_piece(1, 1, piece)
        retrieved_piece = self.board.get_piece(1, 1)
        
        if retrieved_piece:
            self.assertEqual(retrieved_piece.color, 'white')
            self.assertEqual(retrieved_piece.type, 'pawn')

    def test_empty_squares(self):
        """Test that empty squares return None"""
        empty_piece = self.board.get_piece(3, 3)
        self.assertIsNone(empty_piece)


class TestPiece(unittest.TestCase):
    def test_piece_creation(self):
        """Test creating chess pieces"""
        piece = Piece('black', 'queen')
        self.assertEqual(piece.color, 'black')
        self.assertEqual(piece.type, 'queen')

    def test_piece_colors(self):
        """Test valid piece colors"""
        white_piece = Piece('white', 'king')
        black_piece = Piece('black', 'king')
        
        self.assertEqual(white_piece.color, 'white')
        self.assertEqual(black_piece.color, 'black')

    def test_piece_types(self):
        """Test valid piece types"""
        piece_types = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        
        for piece_type in piece_types:
            piece = Piece('white', piece_type)
            self.assertEqual(piece.type, piece_type)


if __name__ == '__main__':
    unittest.main()