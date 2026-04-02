from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chess_ai.utils.chess_encoding import fen_to_planes_cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute planes column for parquet shards.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Source parquet dir with fen, move_id")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output parquet dir with planes, move_id")
    parser.add_argument("--batch-rows", type=int, default=16384, help="Rows per conversion batch")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Parallel worker count")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def convert_file(input_path: Path, output_path: Path, batch_rows: int) -> None:
    pf = pq.ParquetFile(input_path)
    names = set(pf.schema_arrow.names)
    if "fen" not in names or "move_id" not in names:
        raise ValueError(f"Missing required columns in {input_path.name}")

    keep_result = "result" in names
    keep_ply = "ply" in names

    writer: pq.ParquetWriter | None = None
    try:
        for record_batch in pf.iter_batches(batch_size=batch_rows):
            fens = record_batch.column("fen").to_pylist()
            move_ids = np.asarray(record_batch.column("move_id").to_pylist(), dtype=np.int64)

            planes = np.stack([fen_to_planes_cached(fen) for fen in fens], axis=0).astype(np.float32, copy=False)
            flat_values = pa.array(planes.reshape(-1), type=pa.float32())
            planes_arr = pa.FixedSizeListArray.from_arrays(flat_values, 18 * 8 * 8)

            cols = {
                "planes": planes_arr,
                "move_id": pa.array(move_ids),
            }
            if keep_result:
                cols["result"] = record_batch.column("result")
            if keep_ply:
                cols["ply"] = record_batch.column("ply")

            table = pa.table(cols)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    files = sorted(args.input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for input_path in files:
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            continue
        jobs.append((input_path, output_path))

    if not jobs:
        print(f"Done. Precomputed shards are in: {args.output_dir}")
        return

    if args.workers <= 1 or len(jobs) == 1:
        for input_path, output_path in tqdm(jobs, desc="Shards", unit="file"):
            convert_file(input_path, output_path, args.batch_rows)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(convert_file, input_path, output_path, args.batch_rows) for input_path, output_path in jobs]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Shards", unit="file"):
                future.result()

    print(f"Done. Precomputed shards are in: {args.output_dir}")


if __name__ == "__main__":
    main()
