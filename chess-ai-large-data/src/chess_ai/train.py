from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from chess_ai.model.policy_cnn import ChessPolicyCNN
from chess_ai.utils.chess_encoding import build_action_maps, fen_to_planes_cached


class ParquetPositionDataset(IterableDataset):
    def __init__(
        self,
        parquet_dir: Path,
        batch_rows: int = 4096,
        loader_batch_size: int = 256,
        drop_last: bool = True,
        repeat: bool = False,
    ) -> None:
        super().__init__()
        self.parquet_dir = parquet_dir
        self.batch_rows = batch_rows
        self.loader_batch_size = loader_batch_size
        self.drop_last = drop_last
        self.repeat = repeat

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = ds.dataset(str(self.parquet_dir), format="parquet")
        fragments = list(dataset.get_fragments())
        has_planes = "planes" in dataset.schema.names

        worker = get_worker_info()
        if worker is None:
            selected = fragments
        else:
            selected = fragments[worker.id :: worker.num_workers]

        while True:
            for fragment in selected:
                read_columns = ["move_id", "planes"] if has_planes else ["fen", "move_id"]
                scanner = fragment.scanner(columns=read_columns, batch_size=self.batch_rows)
                for batch in scanner.to_batches():
                    move_ids = np.asarray(batch.column("move_id").to_pylist(), dtype=np.int64)

                    if has_planes:
                        planes_col = batch.column("planes")
                        try:
                            flat = planes_col.values.to_numpy(zero_copy_only=False)
                            planes_batch = np.asarray(flat, dtype=np.float32).reshape(len(planes_col), 18, 8, 8)
                        except Exception:
                            planes_batch = np.asarray(planes_col.to_pylist(), dtype=np.float32).reshape(-1, 18, 8, 8)
                    else:
                        fens = batch.column("fen").to_pylist()
                        planes_batch = np.stack([fen_to_planes_cached(fen) for fen in fens], axis=0)

                    n = len(move_ids)
                    step = self.loader_batch_size
                    for i in range(0, n, step):
                        j = min(i + step, n)
                        if self.drop_last and (j - i) < step:
                            continue

                        x_np = planes_batch[i:j]
                        y_np = move_ids[i:j]
                        yield torch.from_numpy(x_np), torch.from_numpy(y_np)

            if not self.repeat:
                break


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_existing_best_valid_loss(checkpoint_dir: Path) -> float:
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        return float("inf")

    try:
        checkpoint = torch.load(best_path, map_location="cpu")
    except Exception:
        return float("inf")

    for key in ("best_valid_loss", "valid_loss"):
        if key in checkpoint and checkpoint[key] is not None:
            try:
                return float(checkpoint[key])
            except (TypeError, ValueError):
                continue
    return float("inf")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: Optional[float],
    max_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    scaler: Optional[GradScaler],
) -> float:
    model.train()
    running_loss = 0.0
    step = 0

    pbar = tqdm(total=max_steps, desc="Train", unit="step")

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader, start=1):
        if step > max_steps:
            break

        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)

        loss_for_backward = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = step % grad_accum_steps == 0 or step == max_steps
        if should_step:
            if scaler is not None:
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss / step:.4f}")
        pbar.update(1)

    pbar.close()
    return running_loss / max(1, min(step, max_steps))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    steps_done = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(loader, start=1):
            if step > max_steps:
                break

            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.numel()
            total_loss += loss.item()
            steps_done += 1

    return {
        "loss": total_loss / max(1, steps_done),
        "acc": total_correct / max(1, total_count),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    out_dir: Path,
    scaler: Optional[GradScaler] = None,
    filename: Optional[str] = None,
    extra_state: Optional[Dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (filename if filename else f"epoch_{epoch:03d}.pt")
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def build_loader(
    parquet_dir: Path,
    batch_size: int,
    num_workers: int,
    batch_rows: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    windows_worker_cap: Optional[int] = 0,
    repeat: bool = False,
) -> DataLoader:
    if os.name == "nt" and num_workers > 0 and windows_worker_cap is not None:
        capped = max(0, min(num_workers, int(windows_worker_cap)))
        if capped != num_workers:
            print(
                f"[warn] Windows worker spawn is memory-heavy; capping num_workers from {num_workers} to {capped}."
            )
        num_workers = capped

    dataset = ParquetPositionDataset(
        parquet_dir=parquet_dir,
        batch_rows=batch_rows,
        loader_batch_size=batch_size,
        drop_last=True,
        repeat=repeat,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**loader_kwargs)


def resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chess move policy model.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_base.yaml"))
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Checkpoint path to resume, 'auto' for latest in checkpoint_dir, 'none' to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    torch.set_float32_matmul_precision("high")

    train_dir = Path(cfg["data"]["train_dir"])
    valid_dir = Path(cfg["data"].get("valid_dir", "")) if cfg["data"].get("valid_dir") else None

    if not train_dir.exists():
        raise FileNotFoundError(f"Train parquet directory not found: {train_dir}")

    _, action_list = build_action_maps()
    num_actions = len(action_list)
    if num_actions <= 1:
        raise RuntimeError("Invalid action space")

    device_name = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    device = torch.device(device_name)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
    print(f"Action space size: {num_actions}")

    use_amp = bool(cfg["train"].get("use_amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(cfg["train"].get("amp_dtype", "bf16"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))

    model = ChessPolicyCNN(
        in_channels=18,
        num_actions=num_actions,
        trunk_channels=cfg["model"]["trunk_channels"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
    ).to(device)

    if bool(cfg["train"].get("channels_last", True)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    compile_enabled = False
    if compile_enabled and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"[warn] torch.compile disabled: {exc}")

    pin_memory = device.type == "cuda"

    train_loader = build_loader(
        train_dir,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        batch_rows=cfg["data"]["reader_batch_rows"],
        pin_memory=pin_memory,
        prefetch_factor=cfg["train"].get("prefetch_factor", 4),
        persistent_workers=cfg["train"].get("persistent_workers", True),
        windows_worker_cap=cfg["train"].get("windows_worker_cap", 0),
        repeat=True,
    )

    valid_loader = None
    if valid_dir and valid_dir.exists():
        valid_loader = build_loader(
            valid_dir,
            batch_size=cfg["train"]["batch_size"],
            num_workers=max(1, cfg["train"]["num_workers"] // 2),
            batch_rows=cfg["data"]["reader_batch_rows"],
            pin_memory=pin_memory,
            prefetch_factor=max(2, cfg["train"].get("prefetch_factor", 4) // 2),
            persistent_workers=cfg["train"].get("persistent_workers", True),
            windows_worker_cap=cfg["train"].get("windows_worker_cap", 0),
            repeat=False,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    steps_per_epoch = cfg["train"]["steps_per_epoch"]
    eval_steps = cfg["train"]["eval_steps"]
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    scaler = GradScaler("cuda") if device.type == "cuda" and use_amp and amp_dtype == torch.float16 else None

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    start_epoch = 1
    best_valid_loss = load_existing_best_valid_loss(checkpoint_dir)
    if best_valid_loss < float("inf"):
        print(f"Existing best valid loss: {best_valid_loss:.4f}")
    resume_arg = (args.resume or "auto").strip().lower()
    resume_path: Optional[Path]
    if resume_arg == "none":
        resume_path = None
    elif resume_arg == "auto":
        resume_path = find_latest_checkpoint(checkpoint_dir)
    else:
        resume_path = Path(args.resume)

    if resume_path is not None and resume_path.exists():
        print(f"Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        last_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        print(f"Resume start epoch: {start_epoch}")

    end_epoch = start_epoch + epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\nEpoch {epoch}/{end_epoch}")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg["optim"].get("grad_clip"),
            max_steps=steps_per_epoch,
            amp_enabled=use_amp,
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            scaler=scaler,
        )
        print(f"Train loss: {train_loss:.4f}")

        if valid_loader is not None:
            metrics = evaluate(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                max_steps=eval_steps,
                amp_enabled=use_amp,
                amp_dtype=amp_dtype,
            )
            print(f"Valid loss: {metrics['loss']:.4f} | Valid acc: {metrics['acc']:.4f}")

            if metrics["loss"] < best_valid_loss:
                best_valid_loss = float(metrics["loss"])
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    checkpoint_dir,
                    scaler,
                    filename="best.pt",
                    extra_state={
                        "valid_loss": float(metrics["loss"]),
                        "valid_acc": float(metrics["acc"]),
                        "best_valid_loss": float(best_valid_loss),
                    },
                )
                print(f"New best checkpoint at epoch {epoch} (valid loss {best_valid_loss:.4f})")

        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, scaler)


if __name__ == "__main__":
    main()
