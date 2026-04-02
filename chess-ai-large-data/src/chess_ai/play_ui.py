from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import pygame
import torch
import yaml

from chess_ai.model.policy_cnn import ChessPolicyCNN
from chess_ai.utils.chess_encoding import build_action_maps, fen_to_planes


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    # Prefer the most recently trained checkpoint even if epoch numbering restarts.
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def find_preferred_checkpoint(checkpoint_dir: Path) -> Path:
    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        return best_ckpt
    return find_latest_checkpoint(checkpoint_dir)


def infer_model_shape_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "stem.0.weight" not in state_dict:
        raise KeyError("Invalid checkpoint: missing stem.0.weight")

    trunk_channels = int(state_dict["stem.0.weight"].shape[0])

    pattern = re.compile(r"res_blocks\.(\d+)\.conv1\.weight")
    block_ids = set()
    for key in state_dict:
        match = pattern.match(key)
        if match:
            block_ids.add(int(match.group(1)))

    num_res_blocks = max(block_ids) + 1 if block_ids else 0
    return trunk_channels, num_res_blocks


def pick_ai_move(
    model: ChessPolicyCNN,
    board: chess.Board,
    move_to_id: Dict[str, int],
    action_list: List[str],
    device: torch.device,
) -> Optional[chess.Move]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    x_np = fen_to_planes(board.fen())
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)[0]

    legal_pairs: List[Tuple[chess.Move, int]] = []
    for move in legal_moves:
        move_id = move_to_id.get(move.uci(), 0)
        if move_id > 0:
            legal_pairs.append((move, move_id))

    if not legal_pairs:
        return legal_moves[np.random.randint(0, len(legal_moves))]

    ids = torch.tensor([idx for _, idx in legal_pairs], dtype=torch.long, device=device)
    legal_logits = logits.index_select(0, ids)
    best_idx = int(torch.argmax(legal_logits).item())
    best_move_id = int(ids[best_idx].item())
    best_uci = action_list[best_move_id]

    try:
        candidate = chess.Move.from_uci(best_uci)
        if candidate in legal_moves:
            return candidate
    except ValueError:
        pass

    return legal_pairs[best_idx][0]


def side_to_color(side: str) -> bool:
    return chess.WHITE if side == "White" else chess.BLACK


def load_inference_bundle(config_path: str, checkpoint_override: Optional[str]) -> Dict:
    cfg = load_yaml(Path(config_path))
    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"]) if checkpoint_override is None else Path(".")

    if checkpoint_override:
        ckpt_path = Path(checkpoint_override)
    else:
        ckpt_path = find_preferred_checkpoint(checkpoint_dir)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state"]

    _, action_list = build_action_maps()
    move_to_id = {move: idx for idx, move in enumerate(action_list)}
    trunk_channels, num_res_blocks = infer_model_shape_from_state_dict(state_dict)

    model = ChessPolicyCNN(
        in_channels=18,
        num_actions=len(action_list),
        trunk_channels=trunk_channels,
        num_res_blocks=num_res_blocks,
    )
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return {
        "model": model,
        "move_to_id": move_to_id,
        "action_list": action_list,
        "device": device,
        "checkpoint": str(ckpt_path),
    }


PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


@dataclass
class Button:
    label: str
    rect: pygame.Rect


class ChessDesktopUI:
    def __init__(self, bundle: Dict, user_side: str = "White") -> None:
        self.bundle = bundle
        self.board = chess.Board()
        self.user_side = user_side
        self.user_color = side_to_color(user_side)
        self.flipped = self.user_side == "Black"

        self.selected_square: Optional[int] = None
        self.legal_targets: List[int] = []
        self.promotion_choice = "q"
        self.status = "Your move"

        self.board_origin = (42, 42)
        self.square_size = 84
        self.board_size = self.square_size * 8
        self.side_panel_x = self.board_origin[0] + self.board_size + 26
        self.window_size = (self.side_panel_x + 340, self.board_origin[1] * 2 + self.board_size)

        self.anim_move: Optional[chess.Move] = None
        self.anim_piece_symbol: Optional[str] = None
        self.anim_start = 0.0
        self.anim_duration = 0.22
        self.last_move: Optional[chess.Move] = None

        self.ai_think_delay = 0.14
        self.ai_ready_at = 0.0

        self.buttons: List[Button] = []

    def reset_game(self) -> None:
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.anim_move = None
        self.status = "New game started"
        self.ai_ready_at = time.perf_counter() + self.ai_think_delay

    def undo_full_turn(self) -> None:
        if self.anim_move is not None:
            return
        if self.board.move_stack:
            self.board.pop()
            if self.board.move_stack:
                self.board.pop()
            self.selected_square = None
            self.legal_targets = []
            self.status = "Undid one full turn"

    def toggle_flip(self) -> None:
        self.flipped = not self.flipped

    def set_side(self, user_side: str) -> None:
        if user_side == self.user_side:
            return
        self.user_side = user_side
        self.user_color = side_to_color(user_side)
        self.flipped = self.user_side == "Black"
        self.reset_game()

    def square_to_screen(self, square: int) -> Tuple[int, int]:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if self.flipped:
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx
        x = self.board_origin[0] + col * self.square_size
        y = self.board_origin[1] + row * self.square_size
        return x, y

    def screen_to_square(self, x: int, y: int) -> Optional[int]:
        bx, by = self.board_origin
        if x < bx or y < by or x >= bx + self.board_size or y >= by + self.board_size:
            return None
        col = (x - bx) // self.square_size
        row = (y - by) // self.square_size
        if self.flipped:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row
        return chess.square(int(file_idx), int(rank_idx))

    def _promotion_for_move(self, from_sq: int, to_sq: int) -> Optional[chess.Move]:
        candidates = [m for m in self.board.legal_moves if m.from_square == from_sq and m.to_square == to_sq]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        promotion_map = {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
        }
        preferred = promotion_map.get(self.promotion_choice, chess.QUEEN)
        for move in candidates:
            if move.promotion == preferred:
                return move
        return candidates[0]

    def _start_animation(self, move: chess.Move) -> None:
        piece = self.board.piece_at(move.from_square)
        self.anim_piece_symbol = piece.symbol() if piece else None
        self.anim_move = move
        self.anim_start = time.perf_counter()

    def _finish_animation_if_needed(self) -> None:
        if self.anim_move is None:
            return
        if time.perf_counter() - self.anim_start < self.anim_duration:
            return
        self.board.push(self.anim_move)
        self.last_move = self.anim_move
        self.anim_move = None
        self.anim_piece_symbol = None
        self.selected_square = None
        self.legal_targets = []
        if self.board.is_game_over():
            self.status = f"Game over: {self.board.result()}"
        else:
            self.status = "AI thinking..." if self.board.turn != self.user_color else "Your move"
            if self.board.turn != self.user_color:
                self.ai_ready_at = time.perf_counter() + self.ai_think_delay

    def handle_board_click(self, square: int) -> None:
        if self.anim_move is not None or self.board.is_game_over() or self.board.turn != self.user_color:
            return

        piece = self.board.piece_at(square)
        if self.selected_square is None:
            if piece and piece.color == self.user_color:
                self.selected_square = square
                self.legal_targets = [m.to_square for m in self.board.legal_moves if m.from_square == square]
                self.status = f"Selected {chess.square_name(square)}"
            else:
                self.status = "Select one of your pieces"
            return

        if square == self.selected_square:
            self.selected_square = None
            self.legal_targets = []
            self.status = "Selection cleared"
            return

        if piece and piece.color == self.user_color and square not in self.legal_targets:
            self.selected_square = square
            self.legal_targets = [m.to_square for m in self.board.legal_moves if m.from_square == square]
            self.status = f"Selected {chess.square_name(square)}"
            return

        move = self._promotion_for_move(self.selected_square, square)
        if move is None:
            self.status = "Illegal destination"
            return
        self._start_animation(move)
        self.status = f"Played {move.uci()}"

    def maybe_ai_turn(self) -> None:
        if self.anim_move is not None or self.board.is_game_over() or self.board.turn == self.user_color:
            return
        if time.perf_counter() < self.ai_ready_at:
            return

        move = pick_ai_move(
            model=self.bundle["model"],
            board=self.board,
            move_to_id=self.bundle["move_to_id"],
            action_list=self.bundle["action_list"],
            device=self.bundle["device"],
        )
        if move is not None:
            self._start_animation(move)
            self.status = f"AI played {move.uci()}"

    def _draw_board_background(self, screen: pygame.Surface) -> None:
        light = pygame.Color("#f2d9b2")
        dark = pygame.Color("#c98b47")
        border = pygame.Color("#111827")
        pygame.draw.rect(
            screen,
            border,
            pygame.Rect(self.board_origin[0] - 8, self.board_origin[1] - 8, self.board_size + 16, self.board_size + 16),
            border_radius=10,
        )
        for rank in range(8):
            for file_idx in range(8):
                square = chess.square(file_idx, rank)
                x, y = self.square_to_screen(square)
                color = light if (rank + file_idx) % 2 == 0 else dark
                pygame.draw.rect(screen, color, pygame.Rect(x, y, self.square_size, self.square_size))

    def _draw_highlights(self, screen: pygame.Surface) -> None:
        if self.last_move is not None:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self.square_to_screen(sq)
                overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                overlay.fill((56, 189, 248, 90))
                screen.blit(overlay, (x, y))

        if self.selected_square is not None:
            x, y = self.square_to_screen(self.selected_square)
            pygame.draw.rect(screen, pygame.Color("#22c55e"), pygame.Rect(x + 3, y + 3, self.square_size - 6, self.square_size - 6), 4)
            for target in self.legal_targets:
                tx, ty = self.square_to_screen(target)
                center = (tx + self.square_size // 2, ty + self.square_size // 2)
                pygame.draw.circle(screen, (15, 23, 42, 170), center, self.square_size // 8)

    def _draw_coordinates(self, screen: pygame.Surface, font_small: pygame.font.Font) -> None:
        coord_color = pygame.Color("#e5e7eb")
        for idx in range(8):
            file_idx = 7 - idx if self.flipped else idx
            label = chr(ord("a") + file_idx)
            text = font_small.render(label, True, coord_color)
            x = self.board_origin[0] + idx * self.square_size + self.square_size // 2 - text.get_width() // 2
            y = self.board_origin[1] + self.board_size + 8
            screen.blit(text, (x, y))

        for idx in range(8):
            rank = idx + 1 if self.flipped else 8 - idx
            text = font_small.render(str(rank), True, coord_color)
            x = self.board_origin[0] - 20
            y = self.board_origin[1] + idx * self.square_size + self.square_size // 2 - text.get_height() // 2
            screen.blit(text, (x, y))

    def _draw_pieces(self, screen: pygame.Surface, piece_font: pygame.font.Font) -> None:
        skip_square = self.anim_move.from_square if self.anim_move else None
        for square, piece in self.board.piece_map().items():
            if square == skip_square:
                continue
            x, y = self.square_to_screen(square)
            symbol = PIECE_UNICODE.get(piece.symbol(), piece.symbol())
            text = piece_font.render(symbol, True, pygame.Color("#111111") if piece.color == chess.BLACK else pygame.Color("#f8fafc"))
            screen.blit(
                text,
                (
                    x + self.square_size // 2 - text.get_width() // 2,
                    y + self.square_size // 2 - text.get_height() // 2 - 2,
                ),
            )

        if self.anim_move is not None and self.anim_piece_symbol is not None:
            start_x, start_y = self.square_to_screen(self.anim_move.from_square)
            end_x, end_y = self.square_to_screen(self.anim_move.to_square)
            t = min(1.0, (time.perf_counter() - self.anim_start) / self.anim_duration)
            ease = 1 - (1 - t) * (1 - t)
            px = start_x + (end_x - start_x) * ease
            py = start_y + (end_y - start_y) * ease
            symbol = PIECE_UNICODE.get(self.anim_piece_symbol, self.anim_piece_symbol)
            piece_obj = chess.Piece.from_symbol(self.anim_piece_symbol)
            text = piece_font.render(symbol, True, pygame.Color("#111111") if piece_obj.color == chess.BLACK else pygame.Color("#f8fafc"))
            screen.blit(
                text,
                (
                    px + self.square_size // 2 - text.get_width() // 2,
                    py + self.square_size // 2 - text.get_height() // 2 - 2,
                ),
            )

    def _draw_side_panel(self, screen: pygame.Surface, title_font: pygame.font.Font, body_font: pygame.font.Font) -> None:
        panel = pygame.Rect(self.side_panel_x, 24, self.window_size[0] - self.side_panel_x - 24, self.window_size[1] - 48)
        pygame.draw.rect(screen, pygame.Color("#0f172a"), panel, border_radius=14)
        pygame.draw.rect(screen, pygame.Color("#1e293b"), panel, 2, border_radius=14)

        y = panel.y + 18
        title = title_font.render("Chess AI Desktop", True, pygame.Color("#f8fafc"))
        screen.blit(title, (panel.x + 16, y))
        y += 44

        lines = [
            f"Checkpoint: {Path(self.bundle['checkpoint']).name}",
            f"Device: {self.bundle['device']}",
            f"You play: {self.user_side}",
            f"Promotion: {self.promotion_choice.upper()}",
            "",
            self.status,
            "",
            "Keys:",
            "N New game | U Undo",
            "F Flip board | W/B Set side",
            "1-4 Promotion: Q R B N",
            "ESC Quit",
        ]
        for line in lines:
            color = pygame.Color("#93c5fd") if line.startswith("Keys") else pygame.Color("#cbd5e1")
            txt = body_font.render(line, True, color)
            screen.blit(txt, (panel.x + 16, y))
            y += 24

        move_title = body_font.render("Recent moves:", True, pygame.Color("#93c5fd"))
        screen.blit(move_title, (panel.x + 16, y + 6))
        y += 34

        pgn_lines: List[str] = []
        for idx, mv in enumerate(self.board.move_stack):
            if idx % 2 == 0:
                pgn_lines.append(f"{idx // 2 + 1}. {mv.uci()}")
            else:
                pgn_lines[-1] = f"{pgn_lines[-1]} {mv.uci()}"

        for line in pgn_lines[-12:]:
            txt = body_font.render(line, True, pygame.Color("#e2e8f0"))
            screen.blit(txt, (panel.x + 16, y))
            y += 21

        self.buttons = [
            Button("New", pygame.Rect(panel.x + 16, panel.bottom - 52, 88, 34)),
            Button("Undo", pygame.Rect(panel.x + 110, panel.bottom - 52, 88, 34)),
            Button("Flip", pygame.Rect(panel.x + 204, panel.bottom - 52, 88, 34)),
        ]
        for btn in self.buttons:
            pygame.draw.rect(screen, pygame.Color("#1d4ed8"), btn.rect, border_radius=8)
            pygame.draw.rect(screen, pygame.Color("#60a5fa"), btn.rect, 2, border_radius=8)
            txt = body_font.render(btn.label, True, pygame.Color("#eff6ff"))
            screen.blit(
                txt,
                (
                    btn.rect.x + btn.rect.width // 2 - txt.get_width() // 2,
                    btn.rect.y + btn.rect.height // 2 - txt.get_height() // 2,
                ),
            )

    def draw(self, screen: pygame.Surface, title_font: pygame.font.Font, piece_font: pygame.font.Font, body_font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        screen.fill(pygame.Color("#020617"))

        # Soft radial glow to make the UI feel less flat.
        glow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        pygame.draw.circle(glow, (59, 130, 246, 48), (self.board_origin[0] + self.board_size // 2, self.board_origin[1] + self.board_size // 2), 340)
        screen.blit(glow, (0, 0))

        self._draw_board_background(screen)
        self._draw_highlights(screen)
        self._draw_pieces(screen, piece_font)
        self._draw_coordinates(screen, small_font)
        self._draw_side_panel(screen, title_font, body_font)


def app(config_path: str, checkpoint: Optional[str], user_side: str = "White") -> None:
    pygame.init()
    pygame.display.set_caption("Chess AI Desktop UI")

    bundle = load_inference_bundle(config_path=config_path, checkpoint_override=checkpoint)
    ui = ChessDesktopUI(bundle=bundle, user_side=user_side)

    screen = pygame.display.set_mode(ui.window_size)
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
    body_font = pygame.font.SysFont("Segoe UI", 20)
    small_font = pygame.font.SysFont("Consolas", 18)
    piece_font = pygame.font.SysFont("Segoe UI Symbol", int(ui.square_size * 0.78))

    running = True
    ui.ai_ready_at = time.perf_counter() + ui.ai_think_delay
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    ui.reset_game()
                elif event.key == pygame.K_u:
                    ui.undo_full_turn()
                elif event.key == pygame.K_f:
                    ui.toggle_flip()
                elif event.key == pygame.K_w:
                    ui.set_side("White")
                elif event.key == pygame.K_b:
                    ui.set_side("Black")
                elif event.key == pygame.K_1:
                    ui.promotion_choice = "q"
                    ui.status = "Promotion set to Queen"
                elif event.key == pygame.K_2:
                    ui.promotion_choice = "r"
                    ui.status = "Promotion set to Rook"
                elif event.key == pygame.K_3:
                    ui.promotion_choice = "b"
                    ui.status = "Promotion set to Bishop"
                elif event.key == pygame.K_4:
                    ui.promotion_choice = "n"
                    ui.status = "Promotion set to Knight"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                hit_button = False
                for btn in ui.buttons:
                    if btn.rect.collidepoint(mx, my):
                        hit_button = True
                        if btn.label == "New":
                            ui.reset_game()
                        elif btn.label == "Undo":
                            ui.undo_full_turn()
                        elif btn.label == "Flip":
                            ui.toggle_flip()
                        break
                if not hit_button:
                    sq = ui.screen_to_square(mx, my)
                    if sq is not None:
                        ui.handle_board_click(sq)

        ui._finish_animation_if_needed()
        ui.maybe_ai_turn()
        ui.draw(screen, title_font, piece_font, body_font, small_font)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against trained Chess AI model with desktop UI")
    parser.add_argument("--config", default="configs/train_base.yaml", help="Path to train config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override")
    parser.add_argument("--side", default="White", choices=["White", "Black"], help="Which side you play")
    args = parser.parse_args()

    app(config_path=args.config, checkpoint=args.checkpoint, user_side=args.side)
