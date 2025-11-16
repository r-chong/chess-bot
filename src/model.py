import numpy as np
import chess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset

import modal

# CONSTANTS
C_PLANES = 12
NUM_MOVES = 64 * 64
NUM_EPOCHS = 15  # Increased from 5 for better convergence

# HuggingFace calls a row an "example"
def preprocess(example):
    board = chess.Board(example["fen"])
    example["input"] = board_to_tensor(board)
    return example

# turns individual tensors into a tensor batch
def collate_fn(batch):
    inputs = torch.stack([b["input"] for b in batch]) # (Batch size, C, 8, 8)

    policy = torch.tensor(
        [b["policy"] for b in batch],
        dtype=torch.long
    )

    value = torch.tensor(
        [b["value"] for b in batch],
        dtype=torch.float32
    ).unsqueeze(1)

    return inputs, policy, value

# for value we want to know who's winning
def make_value_label(example):
    # white centipawns
    cp = example["cp"]
    mate = example["mate"]

    if mate is not None:
        # not None - so if it's mate
        value = 1.0 if mate > 0 else -1.0
    else:
        # squash via tanh
        value = np.tanh(cp / 400.0)

    example["value"] = np.float32(value)
    return example

# extract best move from dataset since we want to know the best move
def make_policy_label(example):
    # line is the best move sequence Stockfish found from that position, written in UCI move format.
    line = example["line"]
    if not line:
        example["policy"] = -1
        return example

    first_move_str = line.split()[0]
    move = chess.Move.from_uci(first_move_str)

    from_sq = move.from_square
    to_sq = move.to_square
    idx = from_sq * 64 + to_sq

    example["policy"] = idx
    return example

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
            nn.Linear(64 * 8 * 8, moves),
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


def load_model(checkpoint_path, device=None):
    """Load a saved model checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleChessNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return model


def main(volume_path=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # collect data
    dset = load_dataset("Lichess/chess-position-evaluations", split="train[:200000]")

    # filter out low-depth / unusable samples
    filtered = dset.filter(lambda row: row["depth"] >= 16 and row["cp"] is not None)
    small_subset = filtered.select(range(100_000))

    tensor_dataset = small_subset.map(preprocess)

    # add new columns based on extracted info from these functions
    tensor_dataset = tensor_dataset.map(make_value_label)
    tensor_dataset = tensor_dataset.map(make_policy_label)

    # remove string columns so it's only numbers
    tensor_dataset = tensor_dataset.remove_columns(["fen", "line"])

    # turn into pytorch tensor format
    tensor_dataset = tensor_dataset.with_format("torch")

    loader = DataLoader(
        tensor_dataset,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True
    )

    model = SimpleChessNet().to(DEVICE)

    policy_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    value_criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # -------------
    # TRAINING LOOP
    # -------------

    # SEE TOP FOR EPOCH CONSTANT
    
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_policy_loss = 0.0
        running_value_loss = 0.0
        n_examples = 0

        for inputs, policy_target, value_target in loader:
            inputs = inputs.to(DEVICE)
            policy_target = policy_target.to(DEVICE)
            value_target = value_target.to(DEVICE)

            optimizer.zero_grad()

            policy_logits, value_pred = model(inputs)

            # Loss
            # the target is the optimal solution and the logits/pred are what we have
            loss_policy = policy_criterion(policy_logits, policy_target)
            loss_value = value_criterion(value_pred, value_target)

            # policy is the probability distribution of potential moves, value is how good our current position is (distilled by stockfish)
            loss = loss_policy + loss_value

            # backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # gradient descent
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_policy_loss += loss_policy.item() * batch_size
            running_value_loss += loss_value.item() * batch_size
            n_examples += batch_size

        epoch_loss = running_loss / n_examples
        epoch_policy = running_policy_loss / n_examples
        epoch_value = running_value_loss / n_examples

        print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} "
                f"- loss: {epoch_loss:.4f} "
                f"(policy: {epoch_policy:.4f}, value: {epoch_value:.4f})"
            )
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        
        # Save checkpoint if loss improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if volume_path:
                checkpoint_path = f"{volume_path}/best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, checkpoint_path)
                print(f"✓ Saved best model checkpoint (loss: {epoch_loss:.4f})")
        
        # Save periodic checkpoint every 5 epochs
        if volume_path and (epoch + 1) % 5 == 0:
            checkpoint_path = f"{volume_path}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    if volume_path:
        final_path = f"{volume_path}/final_model.pt"
        torch.save({
            'epoch': NUM_EPOCHS - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, final_path)
        print(f"✓ Training complete! Final model saved to {final_path}")
    
    return model

##############################################
app = modal.App("chess-eval-training")

# Create persistent volume for model checkpoints
volume = modal.Volume.from_name("chess-models", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "python-chess",
        "datasets",
    )
)

@app.function(
    image=image, 
    timeout=60 * 60,
    volumes={"/models": volume},  # Mount volume at /models
    gpu="T4"  # Request GPU for faster training
)
def run_training():
    """Train the chess model and persist to Modal Volume"""
    model = main(volume_path="/models")
    
    # Commit volume changes so they persist
    volume.commit()
    print("✓ Volume committed - all checkpoints persisted!")
    
    return "Training complete and model saved to volume 'chess-models'"


@app.function(
    image=image,
    volumes={"/models": volume}
)
def list_checkpoints():
    """List all saved model checkpoints"""
    import os
    files = os.listdir("/models")
    print("Available checkpoints:")
    for f in files:
        if f.endswith('.pt'):
            filepath = f"/models/{f}"
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    return files


@app.function(
    image=image,
    volumes={"/models": volume}
)
def download_model(checkpoint_name: str = "best_model.pt"):
    """Download a specific checkpoint from the volume"""
    import os
    filepath = f"/models/{checkpoint_name}"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in volume")
    
    with open(filepath, "rb") as f:
        model_bytes = f.read()
    
    print(f"✓ Read {len(model_bytes) / (1024*1024):.2f} MB from {checkpoint_name}")
    return model_bytes
##############################################

if __name__ == "__main__":
    main()