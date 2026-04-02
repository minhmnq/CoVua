from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_BASE = "https://database.lichess.org/standard"


def build_month_url(month: str) -> str:
    return f"{DEFAULT_BASE}/lichess_db_standard_rated_{month}.pgn.zst"


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with out_path.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Lichess PGN zst files.")
    parser.add_argument("--month", type=str, help="Month in YYYY-MM format, ex: 2025-01")
    parser.add_argument("--url", type=str, help="Direct URL to PGN zst file")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.month and not args.url:
        raise ValueError("Provide --month or --url")

    url = args.url if args.url else build_month_url(args.month)
    filename = url.split("/")[-1]
    out_path = args.out / filename

    print(f"Downloading: {url}")
    print(f"Save to: {out_path}")
    download_file(url, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
