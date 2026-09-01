"""Dynamic image tiling for InternViT (final-plan P1).

`load_image_tiles(path, n_tiles)` returns exactly ``(n_tiles, 3, 448, 448)`` so a
batch can be stacked to ``(B, n_tiles, 3, 448, 448)``. The tiling itself is the
InternVL "dynamic high resolution" scheme; we then pad/truncate to a fixed count.
"""
from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = False) -> list[Image.Image]:
    w, h = image.size
    aspect = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tr = _closest_aspect_ratio(aspect, ratios, w, h, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = []
    for i in range(tr[0] * tr[1]):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def encode_tiles(vision_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, T*P, D): every tile through the vision encoder,
    patch tokens concatenated. Used by both training forward and generation so
    the multi-tile path is defined once."""
    b, t = pixel_values.shape[:2]
    vo = vision_model(pixel_values.flatten(0, 1))
    hs = vo.last_hidden_state if hasattr(vo, "last_hidden_state") else (
        vo[0] if isinstance(vo, (tuple, list)) else vo)
    return hs.reshape(b, t * hs.shape[1], hs.shape[2])


def load_image_tiles(image_path: str, n_tiles: int = 1, image_size: int = 448) -> torch.Tensor:
    """Return exactly ``(n_tiles, 3, image_size, image_size)``.

    n_tiles == 1 -> a single resized image (the current training behaviour).
    n_tiles  > 1 -> dynamic tiles + thumbnail, then padded (repeat last) or
    truncated to n_tiles so every sample in a batch has the same tile count.
    """
    transform = build_transform(image_size)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:  # noqa: BLE001 — one unreadable image must not kill a sweep
        return torch.zeros((n_tiles, 3, image_size, image_size), dtype=torch.float32)

    if n_tiles <= 1:
        return transform(image).unsqueeze(0)

    tiles = dynamic_preprocess(image, max_num=n_tiles, image_size=image_size, use_thumbnail=True)
    pv = torch.stack([transform(t) for t in tiles])  # (T, 3, S, S)
    if pv.shape[0] >= n_tiles:
        return pv[:n_tiles]
    pad = pv[-1:].expand(n_tiles - pv.shape[0], -1, -1, -1)
    return torch.cat([pv, pad], dim=0)
