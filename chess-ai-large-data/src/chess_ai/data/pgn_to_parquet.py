from __future__ import annotations

import argparse
import io
import contextlib
from pathlib import Path
from typing import Dict, Iterator, List, TextIO

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard
from tqdm import tqdm

from chess_ai.utils.chess_encoding import build_action_maps, encode_move_uci


def result_to_value(result: str) -> int:
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


def flush_rows(rows: Dict[str, List], out_dir: Path, shard_idx: int) -> int:
    if not rows["fen"]:
        return shard_idx

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(rows)
    shard_path = out_dir / f"train_{shard_idx:06d}.parquet"
    pq.write_table(table, shard_path, compression="zstd")

    for key in rows:
        rows[key].clear()

    return shard_idx + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PGN/PGN.zst to parquet shards.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .pgn or .pgn.zst")
    parser.add_argument("--output", type=Path, default=Path("data/processed/train"), help="Output dir")
    parser.add_argument("--shard-size", type=int, default=250000, help="Rows per parquet shard")
    parser.add_argument("--max-games", type=int, default=0, help="Stop after N games, 0 = all")
    parser.add_argument("--skip-first-games", type=int, default=0, help="Skip first N games before collecting")
    parser.add_argument("--min-white-elo", type=int, default=0, help="Minimum WhiteElo to keep game")
    parser.add_argument("--min-black-elo", type=int, default=0, help="Minimum BlackElo to keep game")
    parser.add_argument("--min-avg-elo", type=int, default=0, help="Minimum average elo of both players")
    parser.add_argument("--min-plies", type=int, default=0, help="Minimum ply count required to keep game")
    return parser.parse_args()


def parse_elo(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def game_passes_filters(game: chess.pgn.Game, args: argparse.Namespace) -> bool:
    white_elo = parse_elo(game.headers.get("WhiteElo", "0"))
    black_elo = parse_elo(game.headers.get("BlackElo", "0"))

    if white_elo < args.min_white_elo:
        return False
    if black_elo < args.min_black_elo:
        return False

    if args.min_avg_elo > 0:
        avg_elo = (white_elo + black_elo) / 2.0
        if avg_elo < args.min_avg_elo:
            return False

    if args.min_plies > 0:
        try:
            ply_count = int(game.end().ply())
        except Exception:
            ply_count = 0
        if ply_count < args.min_plies:
            return False

    return True


@contextlib.contextmanager
def open_pgn_text_stream(input_path: Path) -> Iterator[TextIO]:
    if input_path.suffix.lower() == ".zst":
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with input_path.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                try:
                    yield text_stream
                finally:
                    text_stream.close()
        return

    with input_path.open("r", encoding="utf-8", errors="ignore") as text_stream:
        yield text_stream


def main() -> None:
    args = parse_args()

    move_to_id, _ = build_action_maps()

    rows: Dict[str, List] = {
        "fen": [],
        "move_id": [],
        "result": [],
        "ply": [],
    }

    shard_idx = 0
    game_count = 0
    kept_games = 0
    row_count = 0

    with open_pgn_text_stream(args.input) as text_stream:
        with tqdm(desc="Games", unit="game") as pbar:
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                game_count += 1
                pbar.update(1)

                if game_count <= args.skip_first_games:
                    continue

                if not game_passes_filters(game, args):
                    continue

                kept_games += 1

                result = result_to_value(game.headers.get("Result", "1/2-1/2"))
                board = game.board()

                for ply, move in enumerate(game.mainline_moves()):
                    fen = board.fen()
                    move_id = encode_move_uci(move.uci(), move_to_id)

                    rows["fen"].append(fen)
                    rows["move_id"].append(move_id)
                    rows["result"].append(result)
                    rows["ply"].append(ply)
                    row_count += 1

                    board.push(move)

                    if len(rows["fen"]) >= args.shard_size:
                        shard_idx = flush_rows(rows, args.output, shard_idx)

                if args.max_games > 0 and kept_games >= args.max_games:
                    break

    shard_idx = flush_rows(rows, args.output, shard_idx)

    print(f"Total games: {game_count}")
    print(f"Kept games: {kept_games}")
    print(f"Total positions: {row_count}")
    print(f"Total shards: {shard_idx}")


if __name__ == "__main__":
    main()
