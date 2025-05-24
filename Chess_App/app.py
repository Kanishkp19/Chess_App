# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
import asyncio
from datetime import datetime
import sqlite3
import json
from services.puzzle_generator import PuzzleGenerator
from models.game import ChessGame

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize services
puzzle_generator = PuzzleGenerator()
active_games = {}  # In-memory game storage
connected_users = {}

def init_user_database():
    """Initialize user database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            rating INTEGER DEFAULT 1200,
            games_played INTEGER DEFAULT 0,
            games_won INTEGER DEFAULT 0,
            puzzle_rating INTEGER DEFAULT 1200,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            white_player TEXT NOT NULL,
            black_player TEXT,
            result TEXT,
            moves TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_user_database()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play')
def play():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('play.html')

@app.route('/puzzles')
def puzzles():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('puzzles.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.json.get('username')
        if username:
            # Simple login - create user if doesn't exist
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
            session['username'] = username
            
            # Store or update user in database
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO users (id, username) VALUES (?, ?)
            ''', (user_id, username))
            conn.commit()
            conn.close()
            
            return jsonify({"success": True, "user_id": user_id})
        return jsonify({"success": False, "error": "Username required"})
    
    return render_template('login.html')

@app.route('/api/puzzles/<theme>')
async def get_puzzles(theme):
    """Get puzzles by theme"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    count = request.args.get('count', 10, type=int)
    min_rating = request.args.get('min_rating', 1200, type=int)
    max_rating = request.args.get('max_rating', 1800, type=int)
    
    try:
        puzzles = await puzzle_generator.generate_puzzles(theme, count, min_rating, max_rating)
        return jsonify({"puzzles": puzzles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/puzzle/<puzzle_id>/solve', methods=['POST'])
async def solve_puzzle(puzzle_id):
    """Submit puzzle solution"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.json
    moves = data.get('moves', [])
    time_taken = data.get('time_taken', 0)
    
    # Validate solution
    result = await puzzle_generator.validate_solution(puzzle_id, moves)
    
    # Record attempt
    if result.get('valid'):
        puzzle_generator.record_attempt(
            puzzle_id, 
            session['user_id'], 
            result.get('complete', False),
            time_taken, 
            moves
        )
    
    return jsonify(result)

@app.route('/api/stats')
def get_user_stats():
    """Get user statistics"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    puzzle_stats = puzzle_generator.get_user_stats(session['user_id'])
    
    # Get game stats from database
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT rating, games_played, games_won, puzzle_rating 
        FROM users WHERE id = ?
    ''', (session['user_id'],))
    
    row = cursor.fetchone()
    game_stats = {
        "rating": row[0] if row else 1200,
        "games_played": row[1] if row else 0,
        "games_won": row[2] if row else 0,
        "puzzle_rating": row[3] if row else 1200
    }
    
    conn.close()
    
    return jsonify({
        "puzzle_stats": puzzle_stats,
        "game_stats": game_stats
    })

# Socket.IO Events for Real-time Gameplay

@socketio.on('connect')
def on_connect():
    if 'user_id' in session:
        connected_users[request.sid] = {
            'user_id': session['user_id'],
            'username': session['username']
        }
        emit('connected', {'user_id': session['user_id']})

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in connected_users:
        del connected_users[request.sid]

@socketio.on('find_game')
def on_find_game():
    """Find a game for the user"""
    if 'user_id' not in session:
        emit('error', {'message': 'Not authenticated'})
        return
    
    user_id = session['user_id']
    username = session['username']
    
    # Look for waiting players
    waiting_game = None
    for game_id, game in active_games.items():
        if game.black_player is None and game.white_player != user_id:
            waiting_game = game
            break
    
    if waiting_game:
        # Join existing game
        waiting_game.black_player = user_id
        game_id = waiting_game.game_id
        
        join_room(game_id)
        
        # Notify both players
        socketio.emit('game_found', {
            'game_id': game_id,
            'white_player': waiting_game.white_player,
            'black_player': user_id,
            'fen': waiting_game.board.fen(),
            'your_color': 'black'
        }, room=game_id)
        
        emit('game_found', {
            'game_id': game_id,
            'white_player': waiting_game.white_player,
            'black_player': user_id,
            'fen': waiting_game.board.fen(),
            'your_color': 'white'
        })
    else:
        # Create new game
        game_id = str(uuid.uuid4())
        game = ChessGame(game_id, user_id)
        active_games[game_id] = game
        
        join_room(game_id)
        emit('waiting_for_opponent', {'game_id': game_id})

@socketio.on('make_move')
def on_make_move(data):
    """Handle move in a game"""
    if 'user_id' not in session:
        emit('error', {'message': 'Not authenticated'})
        return
    
    game_id = data.get('game_id')
    move = data.get('move')
    
    if game_id not in active_games:
        emit('error', {'message': 'Game not found'})
        return
    
    game = active_games[game_id]
    result = game.make_move(move, session['user_id'])
    
    if result['success']:
        # Broadcast move to both players
        socketio.emit('move_made', {
            'move': move,
            'fen': result['fen'],
            'status': result['status'],
            'result': result.get('result'),
            'in_check': result.get('in_check', False)
        }, room=game_id)
        
        # If game ended, save to database
        if result['status'] == 'completed':
            save_game_to_database(game)
            # Remove from active games after a delay
            def cleanup_game():
                if game_id in active_games:
                    del active_games[game_id]
            socketio.start_background_task(lambda: socketio.sleep(30) or cleanup_game())
    else:
        emit('move_error', {'error': result['error']})

@socketio.on('resign')
def on_resign(data):
    """Handle player resignation"""
    if 'user_id' not in session:
        emit('error', {'message': 'Not authenticated'})
        return
    
    game_id = data.get('game_id')
    if game_id not in active_games:
        emit('error', {'message': 'Game not found'})
        return
    
    game = active_games[game_id]
    user_id = session['user_id']
    
    # Determine result based on who resigned
    if user_id == game.white_player:
        game.result = 'black_wins'
    else:
        game.result = 'white_wins'
    
    game.status = 'completed'
    
    # Notify both players
    socketio.emit('game_ended', {
        'result': game.result,
        'reason': 'resignation'
    }, room=game_id)
    
    save_game_to_database(game)

def save_game_to_database(game):
    """Save completed game to database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO game_history (game_id, white_player, black_player, result, moves)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        game.game_id,
        game.white_player,
        game.black_player,
        game.result,
        json.dumps(game.moves_history)
    ))
    
    # Update player statistics
    if game.result == 'white_wins':
        winner, loser = game.white_player, game.black_player
    elif game.result == 'black_wins':
        winner, loser = game.black_player, game.white_player
    else:
        winner, loser = None, None
    
    # Update games played for both players
    for player in [game.white_player, game.black_player]:
        if player:
            cursor.execute('''
                UPDATE users SET games_played = games_played + 1
                WHERE id = ?
            ''', (player,))
    
    # Update wins for winner
    if winner:
        cursor.execute('''
            UPDATE users SET games_won = games_won + 1
            WHERE id = ?
        ''', (winner,))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
