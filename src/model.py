import numpy as np
import chess

C_PLANES = 12

def board_to_tensor(board: chess.Board):   
    tensor = np.zeros((8, 8, C_PLANES), dtype=np.float32)

    # what each of the planes represent
    # we separated for all of the types of pieces, and differentiate between white and black by adding 6 later
    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING, 5
    }

    for square, piece in board.piece_map().items():
        # file is x and rank is y
        # we subtract from 7 because coordinate system defaults to (0,0) being top left
        # however we want (0,0) to be the bottom left.
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        idx = piece_to_index[piece.piece_type]

        if piece.color = chess.BLACK:
            idx += 6 # offset for black pieces

        tensor[row, col, idx] = 1

