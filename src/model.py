import numpy as np
import chess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
dset = load_dataset("Lichess/chess-evaluations", split="train")

C_PLANES = 12
NUM_MOVES = 64 * 64

# HuggingFace calls a row an "example"
def preprocess(example):
    board = chess.Board(example["fen"])
    example["input"] = board_to_tensor(board)
    return example

# turns individual tensors into a tensor batch
def collate_fn(batch):
    return torch.stack(b["input"] for b in batch)

def board_to_tensor(board: chess.Board):   
    tensor = np.zeros((C_PLANES, 8, 8), dtype=np.float32)

    # what each of the planes represent
    # we separated for all of the types of pieces, and differentiate between white and black by adding 6 later
    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        # file is x and rank is y
        # we subtract from 7 because chess board defaults to (0,0) being bottom left
        # however we want (0,0) to be the top left.
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        idx = piece_to_index[piece.piece_type]

        if piece.color == chess.BLACK:
            idx += 6 # offset for black pieces

        tensor[idx, row, col] = 1
    
    # return as torch tensor (C, H, W)
    return torch.from_numpy(tensor)
    
# nn.Module parent class is from PyTorch
class SimpleChessNet(nn.Module):
    def __init__(self, moves=NUM_MOVES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(C_PLANES, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.flat = nn.Flatten()

        # predict logits over move indices
        self.policy_head = nn.Sequential(
            # 4096 "possible" moves (not actually)
            nn.Linear(64 * 8 * 8, NUM_MOVES),
        )

        # predict scalar in range [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),

            # activation; squash into [-1, 1]
            nn.Tanh()
        )
    
    def forward(self, x):
        # x is our tensor
        x = self.cnn(x)

        x = self.flat(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


def main():
    # collect data
    filtered = dset.filter(lambda row: row["white_elo"] >= 1000 and row["black_elo"] >= 1000)
    small_subset = filtered.select(range(100_000))
    tensor_dataset = small_subset.map(preprocess, remove_columns=small_subset.column_names)

    # total_loss = cross_entropy(policy, labels_policy) + MSE(value, labels_value)

if __name__ == "__main__":
    main()