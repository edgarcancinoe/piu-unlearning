from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from piu_unlearning.models.arc2face import GeneratedSamples

if TYPE_CHECKING:
    from piu_unlearning.data import EvaluationConditions


FONT = ImageFont.load_default()
COLORS = {"loss": "#202124", "forget_loss": "#d93025", "preserve_loss": "#188038", "forget_ism": "#d93025", "retain_ism": "#1a73e8"}


def load_history(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def draw_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, history: list[dict[str, float]], series: tuple[tuple[str, str], ...]) -> None:
    left, top, right, bottom = box
    values = [entry[key] for entry in history for key, _ in series]
    minimum, maximum = min(values), max(values)
    if minimum == maximum: minimum, maximum = minimum - 1.0, maximum + 1.0
    padding = (maximum - minimum) * 0.05
    minimum, maximum = minimum - padding, maximum + padding
    draw.rectangle(box, outline="#9aa0a6", width=1)
    draw.text((left, top - 18), title, fill="#202124", font=FONT)
    draw.text((left, bottom + 4), f"{history[0]['step']} steps", fill="#5f6368", font=FONT)
    draw.text((right - 95, bottom + 4), f"{history[-1]['step']} steps", fill="#5f6368", font=FONT)
    for index, (key, color) in enumerate(series):
        points = [(left + (right - left) * offset / max(1, len(history) - 1), bottom - (entry[key] - minimum) / (maximum - minimum) * (bottom - top)) for offset, entry in enumerate(history)]
        draw.line(points, fill=color, width=3)
        draw.text((left + index * 125, top + 4), key.replace("_", " "), fill=color, font=FONT)


def write_training_curves(loss_history_path: Path, ism_history_path: Path, output_path: Path, method: str = "piu") -> None:
    loss_history = load_history(loss_history_path)
    ism_history = load_history(ism_history_path) if ism_history_path.exists() else []
    image = Image.new("RGB", (1800, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), f"{method.upper()} training curves", fill="#202124", font=FONT)
    draw_chart(draw, (50, 70, 580, 630), "Total and forget loss", loss_history, (("loss", COLORS["loss"]), ("forget_loss", COLORS["forget_loss"])))
    if method == "wid":
        draw_chart(draw, (635, 70, 1165, 630), "Weighted WID components", loss_history, (("model_term", "#d93025"), ("identity_term", "#1a73e8"), ("preserve_term", "#188038")))
    elif method == "siss": draw_chart(draw, (635, 70, 1165, 630), "Retain diffusion loss", loss_history, (("retain_loss", "#188038"),))
    else: draw_chart(draw, (635, 70, 1165, 630), "Preservation loss", loss_history, (("preserve_loss", COLORS["preserve_loss"]),))
    if ism_history: draw_chart(draw, (1220, 70, 1750, 630), "Validation ISM", ism_history, (("forget_ism", COLORS["forget_ism"]), ("retain_ism", COLORS["retain_ism"])))
    image.save(output_path)


def paste_image(canvas: Image.Image, path: Path, position: tuple[int, int], size: int) -> None:
    with Image.open(path) as image: canvas.paste(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), position)


def write_fixed_condition_grid(before: GeneratedSamples, after: GeneratedSamples, conditions: EvaluationConditions, output_path: Path) -> None:
    tile_size, label_width, header_height, row_height = 128, 250, 50, 148
    image = Image.new("RGB", (label_width + 4 * tile_size, header_height + len(before.forget) * row_height), "white")
    draw = ImageDraw.Draw(image)
    headers = ("Forget before", "Forget after", "Retain before", "Retain after")
    for index, header in enumerate(headers): draw.text((label_width + index * tile_size + 8, 18), header, fill="#202124", font=FONT)
    for index in range(len(before.forget)):
        top = header_height + index * row_height
        label = f"{index:02d}  forget {conditions.forget_labels[index].item()} / seed {conditions.forget_seeds[index]}\nretain {conditions.retain_labels[index].item()} / seed {conditions.retain_seeds[index]}"
        draw.multiline_text((8, top + 44), label, fill="#5f6368", font=FONT, spacing=3)
        paths = (before.forget[index], after.forget[index], before.retain[index], after.retain[index])
        for column, path in enumerate(paths): paste_image(image, path, (label_width + column * tile_size, top), tile_size)
    image.save(output_path)


def create_summary_visuals(before: GeneratedSamples, after: GeneratedSamples, conditions: EvaluationConditions, output_dir: Path, method: str = "piu") -> tuple[Path | None, Path]:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    curves_path, grid_path = summary_dir / "training_curves.png", summary_dir / "fixed_conditions.png"
    if method == "uce": curves_path = None
    else: write_training_curves(output_dir / "checkpoints" / "loss_history.jsonl", output_dir / "checkpoints" / "ism_history.jsonl", curves_path, method)
    write_fixed_condition_grid(before, after, conditions, grid_path)
    return curves_path, grid_path


def write_method_comparison(samples: dict[str, GeneratedSamples], conditions: EvaluationConditions, output_dir: Path) -> None:
    """Compare the same conditions across the original model and edited models."""
    tile, label_width, header, row_height = 128, 200, 40, 148
    for partition in ("forget", "retain"):
        labels, seeds = getattr(conditions, f"{partition}_labels"), getattr(conditions, f"{partition}_seeds")
        canvas = Image.new("RGB", (label_width + tile * len(samples), header + row_height * len(labels)), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (method, images) in enumerate(samples.items()):
            draw.text((label_width + column * tile + 8, 12), method.upper(), fill="#202124", font=FONT)
            for row, path in enumerate(getattr(images, partition)):
                top = header + row * row_height
                if column == 0: draw.text((8, top + 50), f"{row:02d} ID {labels[row].item()} / seed {seeds[row]}", fill="#5f6368", font=FONT)
                paste_image(canvas, path, (label_width + column * tile, top), tile)
        canvas.save(output_dir / f"comparison_{partition}.png")
