"""Stockfish engine management"""

import chess.engine
import asyncio
from utils import find_stockfish, get_engine_limits
from config import (
    STOCKFISH_HASH_SIZE, STOCKFISH_MAX_THREADS, STOCKFISH_PONDER,
    STOCKFISH_CONTEMPT, STOCKFISH_ANALYSE_MODE
)


class EngineManager:
    def __init__(self):
        self.engine = None
    
    async def initialize_engine(self):
        """Initialize Stockfish engine"""
        try:
            stockfish_path = find_stockfish()
            if stockfish_path:
                print(f"Found Stockfish at: {stockfish_path}")
                transport, self.engine = await chess.engine.popen_uci(stockfish_path)
                
                # Configure Stockfish for optimal performance
                await self._configure_stockfish_options()
                
                print("Stockfish engine initialized and configured successfully!")
                return True
            else:
                print("ERROR: Stockfish not found!")
                print("To install Stockfish:")
                print("- Ubuntu/Debian: sudo apt-get install stockfish")
                print("- macOS: brew install stockfish")
                print("- Windows: Download from https://stockfishchess.org/download/")
                raise Exception("Stockfish not found")
                
        except Exception as e:
            print(f"Error initializing engine: {e}")
            raise e
    
    async def _configure_stockfish_options(self):
        """Configure Stockfish for optimal performance"""
        try:
            # Set hash table size (memory for transposition table) - 256MB for better performance
            await self.engine.configure({"Hash": STOCKFISH_HASH_SIZE})
            
            # Set number of threads (use available CPU cores, capped for web server)
            await self.engine.configure({"Threads": STOCKFISH_MAX_THREADS})
            
            # Disable Ponder (thinking on opponent's time) for web app
            await self.engine.configure({"Ponder": STOCKFISH_PONDER})
            
            # Set contempt to 0 (neutral draw preference)
            await self.engine.configure({"Contempt": STOCKFISH_CONTEMPT})
            
            # Set UCI_AnalyseMode to false for game play
            await self.engine.configure({"UCI_AnalyseMode": STOCKFISH_ANALYSE_MODE})
            
            print(f"Stockfish configured: Hash={STOCKFISH_HASH_SIZE}MB, Threads={STOCKFISH_MAX_THREADS}")
            
        except Exception as e:
            print(f"Warning: Could not configure some Stockfish options: {e}")
    
    async def get_engine_move(self, board, difficulty):
        """Get a move from the engine based on difficulty"""
        if not self.engine:
            raise Exception("Engine not initialized")
        
        limits = get_engine_limits(difficulty)
        
        # Configure skill level based on difficulty
        await self.engine.configure({"Skill Level": limits["skill_level"]})
        
        # Create engine limit with both time and depth
        engine_limit = chess.engine.Limit(
            time=limits["time"],
            depth=limits["depth"]
        )
        
        print(f"Engine thinking: Difficulty {difficulty}, Skill {limits['skill_level']}, Time {limits['time']}s, Depth {limits['depth']}")
        
        # Get the move with timeout protection
        timeout = limits["time"] + 10.0  # Extra buffer for engine overhead
        result = await asyncio.wait_for(
            self.engine.play(board, engine_limit),
            timeout=timeout
        )
        
        if not result.move:
            raise Exception("Engine failed to find a move")
        
        return result.move
    
    async def analyze_position(self, board, depth=12, multipv=3):
        """Analyze position and return top moves"""
        if not self.engine:
            raise Exception("Engine not initialized")
        
        # Limit depth for web responsiveness
        depth = min(max(depth, 5), 18)
        
        # Get analysis with timeout protection
        info = await asyncio.wait_for(
            self.engine.analyse(
                board, 
                chess.engine.Limit(depth=depth),
                multipv=multipv
            ),
            timeout=10.0
        )
        
        analysis = []
        for i, pv_info in enumerate(info):
            if "pv" in pv_info and pv_info["pv"]:
                move = pv_info["pv"][0]
                score = pv_info.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                analysis.append({
                    "rank": i + 1,
                    "move": move.uci(),
                    "move_san": board.san(move),
                    "score": str(score.white()),
                    "pv": [m.uci() for m in pv_info["pv"][:5]]  # First 5 moves of variation
                })
        
        return analysis, depth
    
    async def test_engine(self):
        """Test engine functionality"""
        if not self.engine:
            return False, "Engine not initialized"
        
        try:
            test_board = chess.Board()
            info = await self.engine.analyse(test_board, chess.engine.Limit(time=0.1))
            
            return True, {
                "name": "Stockfish",
                "version": "Latest",
                "options_configured": True,
                "analysis_sample": str(info[0].get("score", "N/A"))
            }
        except Exception as e:
            return False, f"Engine test failed: {str(e)}"
    
    async def close_engine(self):
        """Close the engine"""
        if self.engine:
            try:
                await self.engine.quit()
                print("Stockfish engine closed successfully")
            except Exception as e:
                print(f"Error closing engine: {e}")


# Global engine manager instance
engine_manager = EngineManager()