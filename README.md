CHESS AI TRAINING PIPELINE (LARGE DATA)

Muc tieu
- Train model AI co vua du doan nuoc di tiep theo (policy model) tu dataset PGN lon.
- Ho tro file .pgn.zst hang chuc den hang tram GB (Lichess DB).
- Xu ly theo shard Parquet de tranh full load vao RAM.

Kien truc tong quan
1) Download PGN zst tu Lichess.
2) Parse PGN -> vi tri (FEN) + move_id -> luu Parquet shard.
3) Dung DataLoader dang stream doc shard.
4) Train CNN policy model (kieu AlphaZero supervised).

Cau truc project
- src/chess_ai/data/download_lichess.py: download file zst.
- src/chess_ai/data/pgn_to_parquet.py: convert PGN zst sang Parquet.
- src/chess_ai/utils/chess_encoding.py: action space + encode FEN.
- src/chess_ai/model/policy_cnn.py: model CNN + residual blocks.
- src/chess_ai/train.py: train loop + evaluate + checkpoint.
- configs/train_base.yaml: cau hinh train.
- scripts/run_pipeline.ps1: script chay nhanh.

Yeu cau
- Python 3.10+
- GPU CUDA (khuyen nghi), van chay duoc CPU nhung cham.
- Disk lon (data raw + parquet + checkpoint).

Cach chay nhanh (PowerShell)
1) Tao moi truong va cai package
   python -m venv .venv
   $env:PYTHONPATH = "src"
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

   # RTX 50-series (sm_120): dung PyTorch nightly CUDA moi hon
   .\.venv\Scripts\python.exe -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

2) Download 1 thang du lieu Lichess
   .\.venv\Scripts\python.exe -m chess_ai.data.download_lichess --month 2025-01 --out data/raw

3) Convert sang Parquet (thu nghiem 200k game truoc)
   .\.venv\Scripts\python.exe -m chess_ai.data.pgn_to_parquet --input data/raw/lichess_db_standard_rated_2025-01.pgn.zst --output data/processed/train --max-games 200000

4) Train
   .\.venv\Scripts\python.exe -m chess_ai.train --config configs/train_base.yaml

Chay bang script all-in-one
- scripts/run_pipeline.ps1

Meo de train lon
- Tang num_workers len 8 hoac 16 neu CPU du manh.
- Tang batch_size toi muc GPU chiu duoc.
- Dung nhieu thang du lieu: convert tung file .zst vao cung thu muc Parquet train.
- Tach valid set rieng (data/processed/valid) de theo doi overfit.
- Co the bo sung mixed precision, DDP, va wandb de scale tiep.

Luu y quan trong de model manh hon
- Du lieu random trong scripts/gen_sample_pgn.py chi hop de smoke test, khong phai du lieu tot de hoc chess.
- Muon model choi kha hon, nen train tren Lichess that va uu tien dung cac nuoc trong game that.
- Khi da co parquet, chay precompute planes roi train tren data/processed/train_planes de bo bottleneck FEN encoding.
- Neu muon train nhanh ma chat luong tot, dung scripts/run_pipeline.ps1 voi month Lichess va config train_30min_fast.yaml.

Nang cap tiep theo
- Them value head (du doan ket qua tran dau) de huan luyen da muc tieu.
- Fine-tune bang self-play + MCTS.
- Distill tu engine manh (Stockfish labels) de tang chat luong.

UI desktop choi co voi model da train (mo truc tiep tu VS Code)
1) Cai dependencies
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

1-click trong VS Code (nut Play)
- Da co san cau hinh trong .vscode/launch.json.
- Mo tab Run and Debug, chon "Play Chess UI (White)" hoac "Play Chess UI (Black)", bam nut Play (F5).
- UI se mo truc tiep trong cua so desktop, khong can trinh duyet va khong can go lenh.

2) Chay UI desktop (khong can trinh duyet)
   $env:PYTHONPATH = "src"
   .\.venv\Scripts\python.exe -m chess_ai.play_ui

3) Tuy chon checkpoint cu the
   .\.venv\Scripts\python.exe -m chess_ai.play_ui --checkpoint outputs/checkpoints/epoch_010.pt --side White

Tinh nang UI
- Tu dong load checkpoint moi nhat trong outputs/checkpoints neu khong chi dinh.
- Nguoi choi co the danh trang hoac den.
- Bam truc tiep vao o tren ban co de di chuyen: click o nguon roi click o dich.
- Highlight ro o dang chon + cac o dich hop le.
- Co animation nuoc di (nguoi choi va AI), highlight nuoc di vua danh.
- Nut New/Undo/Flip va hotkey nhanh:
  - N: New game
  - U: Undo 1 luot (2 nua nuoc)
  - F: Flip board
  - W/B: Chon phe
  - 1/2/3/4: Promotion Q/R/B/N
