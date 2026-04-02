#!/usr/bin/env python3
"""Generate sample PGN games quickly for smoke testing the entire pipeline."""

import argparse
import random
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm


def generate_random_game() -> str:
    """Generate a single valid chess game."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    # Generate 20-80 moves on average
    max_moves = random.randint(20, 80)
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        move = random.choice(legal_moves)
        node = node.add_variation(move)
        board.push(move)
        move_count += 1

    # Set result
    if board.is_checkmate():
        game.headers["Result"] = "1-0" if board.turn else "0-1"
    elif board.is_stalemate() or board.halfmove_clock >= 100:
        game.headers["Result"] = "1/2-1/2"
    else:
        # Random result if game incomplete
        game.headers["Result"] = random.choice(["1-0", "0-1", "1/2-1/2"])

    # Add headers
    game.headers["Event"] = "Sample Chess Game"
    game.headers["Site"] = "localhost"
    game.headers["Date"] = "2026.03.31"
    game.headers["Round"] = str(random.randint(1, 100))
    game.headers["White"] = f"White{random.randint(1, 9999)}"
    game.headers["Black"] = f"Black{random.randint(1, 9999)}"

    return str(game)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic PGN games.")
    parser.add_argument("--num-games", type=int, default=100_000, help="Number of games to generate")
    parser.add_argument("--out", type=Path, default=Path("data/raw/sample_games.pgn"), help="Output PGN file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    out_file = args.out
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_games:,} random chess games to {out_file}...")

    with out_file.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(args.num_games), desc="Games"):
            pgn_str = generate_random_game()
            f.write(pgn_str)
            f.write("\n\n")

    print(f"✓ Generated {args.num_games:,} games to {out_file}")
    print(f"  File size: {out_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
