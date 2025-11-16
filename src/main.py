from .utils import chess_manager, GameContext
from .inference import ChessEngine
from chess import Move
import time

# Load the trained model once at startup
print("Loading chess model...")
engine = ChessEngine(model_path="best_model-2.pt", device="cpu")
print("Model loaded and ready!")


@chess_manager.entrypoint
def make_move(ctx: GameContext):
    """
    Main entrypoint - called every time the bot needs to make a move
    Returns a python-chess Move object that is a legal move for the current position
    """
    start_time = time.perf_counter()
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Get model prediction
    best_move, move_probs, position_value = engine.predict_move(
        ctx.board,
        temperature=1.0  # Adjust for more/less deterministic play
    )
    
    # Log probabilities for visualization
    ctx.logProbabilities(move_probs)
    
    # Log some useful info
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"Move: {best_move.uci()}")
    print(f"Position eval: {position_value:.3f} ({'White' if position_value > 0 else 'Black'} advantage)")
    print(f"Top 3 moves: {[(m.uci(), f'{p:.2%}') for m, p in list(move_probs.items())[:3]]}")
    print(f"Inference time: {elapsed:.1f}ms")
    
    return best_move


@chess_manager.reset
def reset_game(ctx: GameContext):
    """
    Called when a new game begins
    Can be used to clear caches, reset model state, etc.
    """
    print("New game started - board reset")
    # No state to reset for this stateless model
    pass
