from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import chess
import numpy as np


PROMOTION_PIECES = ("q", "r", "b", "n")


def generate_action_space() -> List[str]:
    actions = set()

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            actions.add(move.uci())

    for file_idx in range(8):
        # White promotions.
        from_sq = chess.square(file_idx, 6)
        to_candidates = [chess.square(file_idx, 7)]
        if file_idx > 0:
            to_candidates.append(chess.square(file_idx - 1, 7))
        if file_idx < 7:
            to_candidates.append(chess.square(file_idx + 1, 7))

        for to_sq in to_candidates:
            base = chess.Move(from_sq, to_sq).uci()
            for piece in PROMOTION_PIECES:
                actions.add(f"{base}{piece}")

        # Black promotions.
        from_sq = chess.square(file_idx, 1)
        to_candidates = [chess.square(file_idx, 0)]
        if file_idx > 0:
            to_candidates.append(chess.square(file_idx - 1, 0))
        if file_idx < 7:
            to_candidates.append(chess.square(file_idx + 1, 0))

        for to_sq in to_candidates:
            base = chess.Move(from_sq, to_sq).uci()
            for piece in PROMOTION_PIECES:
                actions.add(f"{base}{piece}")

    return sorted(actions)


def build_action_maps() -> Tuple[Dict[str, int], List[str]]:
    action_list = ["<unk>"] + generate_action_space()
    move_to_id = {move: idx for idx, move in enumerate(action_list)}
    return move_to_id, action_list


def encode_move_uci(move_uci: str, move_to_id: Dict[str, int]) -> int:
    return move_to_id.get(move_uci, 0)


def fen_to_planes(fen: str) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    try:
        board_part, turn, castling, _ep, halfmove, *_rest = fen.split(" ")
        piece_to_channel = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4,
            "K": 5,
            "p": 6,
            "n": 7,
            "b": 8,
            "r": 9,
            "q": 10,
            "k": 11,
        }

        ranks = board_part.split("/")
        for row_from_top, rank_str in enumerate(ranks):
            rank_idx = 7 - row_from_top
            file_idx = 0
            for ch in rank_str:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    channel = piece_to_channel.get(ch)
                    if channel is not None and 0 <= file_idx < 8:
                        planes[channel, rank_idx, file_idx] = 1.0
                    file_idx += 1

        planes[12, :, :] = 1.0 if turn == "w" else 0.0
        planes[13, :, :] = 1.0 if "K" in castling else 0.0
        planes[14, :, :] = 1.0 if "Q" in castling else 0.0
        planes[15, :, :] = 1.0 if "k" in castling else 0.0
        planes[16, :, :] = 1.0 if "q" in castling else 0.0
        planes[17, :, :] = min(float(halfmove) / 100.0, 1.0)
        return planes
    except Exception:
        # Fallback for unexpected FEN variants.
        board = chess.Board(fen)
        piece_offset = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square)
            file_idx = chess.square_file(square)
            base = 0 if piece.color == chess.WHITE else 6
            channel = base + piece_offset[piece.piece_type]
            planes[channel, rank, file_idx] = 1.0

        planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)
        return planes


@lru_cache(maxsize=300000)
def fen_to_planes_cached(fen: str) -> np.ndarray:
    # Cache common opening and transposition positions to reduce CPU encoding cost.
    return fen_to_planes(fen)
