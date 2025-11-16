"""
Optimized inference engine for the chess model
"""
import torch
import torch.nn as nn
import chess
import numpy as np
from typing import Dict, Tuple, Optional

# CONSTANTS
C_PLANES = 12
NUM_MOVES = 64 * 64


def board_to_tensor(board: chess.Board) -> torch.Tensor:   
    """Convert a chess board to a tensor representation"""
    tensor = np.zeros((C_PLANES, 8, 8), dtype=np.float32)

    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        idx = piece_to_index[piece.piece_type]
        
        if piece.color == chess.BLACK:
            idx += 6  # offset for black pieces
        
        tensor[idx, row, col] = 1
    
    return torch.from_numpy(tensor)


class ChessNetInference(nn.Module):
    """Optimized inference-only chess neural network"""
    
    def __init__(self, moves=NUM_MOVES):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(C_PLANES, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Policy head (move prediction)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, moves),
        )

        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value


class ChessEngine:
    """High-level interface for chess move prediction"""
    
    def __init__(self, model_path: str = "best_model-2.pt", device: Optional[str] = None):
        """
        Initialize the chess engine
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (cuda/cpu). Auto-detected if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = ChessNetInference().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Chess engine loaded from {model_path}")
        print(f"  - Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict_move(
        self, 
        board: chess.Board, 
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[chess.Move, Dict[chess.Move, float], float]:
        """
        Predict the best move for the current board position
        
        Args:
            board: Current chess board state
            temperature: Temperature for softmax (higher = more random, lower = more deterministic)
            top_k: If set, only consider top-k moves
        
        Returns:
            Tuple of (best_move, move_probabilities, position_value)
        """
        # Convert board to tensor
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value_pred = self.model(board_tensor)
        
        # Apply temperature scaling
        policy_logits = policy_logits / temperature
        
        # Get probabilities
        probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        
        # Map to legal moves
        legal_moves = list(board.legal_moves)
        move_probs = {}
        
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            idx = from_sq * 64 + to_sq
            
            if idx < len(probs):
                move_probs[move] = float(probs[idx])
        
        # Normalize probabilities among legal moves
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            # Fallback to uniform distribution if no valid moves found
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}
        
        # Apply top-k filtering if requested
        if top_k is not None and len(move_probs) > top_k:
            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
            top_moves = dict(sorted_moves[:top_k])
            # Re-normalize
            total = sum(top_moves.values())
            move_probs = {move: prob / total for move, prob in top_moves.items()}
        
        # Select best move
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        
        # Get position value
        position_value = float(value_pred[0].item())
        
        return best_move, move_probs, position_value
    
    def get_top_moves(
        self, 
        board: chess.Board, 
        n: int = 5
    ) -> list[Tuple[chess.Move, float]]:
        """
        Get the top N moves with their probabilities
        
        Args:
            board: Current chess board state
            n: Number of top moves to return
        
        Returns:
            List of (move, probability) tuples, sorted by probability
        """
        _, move_probs, _ = self.predict_move(board)
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_moves[:n]
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the current position without making a move
        
        Args:
            board: Current chess board state
        
        Returns:
            Position evaluation (-1 to 1, positive favors white)
        """
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value_pred = self.model(board_tensor)
        
        return float(value_pred[0].item())

