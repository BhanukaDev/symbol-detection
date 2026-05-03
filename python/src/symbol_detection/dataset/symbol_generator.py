"""Symbol-focused dataset generator — no floor plan logic."""

import cv2
import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from effects import water_wave_distortion, twirl_distortion, apply_gradient_fade
    _EFFECTS = True
except ImportError:
    _EFFECTS = False

SYMBOL_CATEGORIES = [
    {"id": 1, "name": "Light"},
    {"id": 2, "name": "Two-pole, one-way switch"},
    {"id": 3, "name": "Duplex Receptacle"},
    {"id": 4, "name": "Junction Box"},
    {"id": 5, "name": "Single-pole, one-way switch"},
    {"id": 6, "name": "Two-way switch"},
    {"id": 7, "name": "Three-pole, one-way switch"},
]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_symbols(symbols_dir: str) -> Dict[str, List[np.ndarray]]:
    result: Dict[str, List[np.ndarray]] = {}
    for folder in Path(symbols_dir).iterdir():
        if not folder.is_dir():
            continue
        variants = [
            cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            for p in folder.glob("*.png")
        ]
        variants = [v for v in variants if v is not None]
        if variants:
            result[folder.name] = variants
    return result


def _resize_to_target(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize so the longer side == target_size, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if angle % 360 == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    border = (0, 0, 0, 0) if img.shape[2] == 4 else (255, 255, 255)
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def _apply_symbol_effects(img: np.ndarray) -> np.ndarray:
    if not _EFFECTS:
        return img
    result = img.copy()
    if random.random() < 0.5:
        if random.random() < 0.5 and water_wave_distortion:
            result = water_wave_distortion(
                result,
                amplitude=random.randint(1, 3),
                frequency=random.uniform(0.005, 0.025),
            )
        elif twirl_distortion:
            result = twirl_distortion(
                result,
                angle=random.uniform(0.03, 0.18),
                radius=min(result.shape[:2]) // 2,
            )
    if random.random() < 0.4 and apply_gradient_fade:
        result = apply_gradient_fade(result,
                                     min_alpha=random.uniform(0.35, 0.7),
                                     max_alpha=1.0)
    return result


def _blend(canvas: np.ndarray, sym: np.ndarray, x: int, y: int) -> None:
    sh, sw = sym.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + sw, canvas.shape[1]), min(y + sh, canvas.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    sc = sym[y1 - y: y2 - y, x1 - x: x2 - x]
    roi = canvas[y1:y2, x1:x2]
    if sc.shape[2] == 4:
        a = sc[:, :, 3:4] / 255.0
        canvas[y1:y2, x1:x2] = (sc[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
    else:
        canvas[y1:y2, x1:x2] = sc[:, :, :3]


def _overlaps(x: int, y: int, w: int, h: int,
              boxes: List[Tuple[int, int, int, int]], pad: int) -> bool:
    for bx, by, bw, bh in boxes:
        if not (x + w + pad <= bx or bx + bw + pad <= x or
                y + h + pad <= by or by + bh + pad <= y):
            return True
    return False


def _make_background(width: int, height: int) -> np.ndarray:
    base = random.randint(228, 255)
    canvas = np.full((height, width, 3), base, dtype=np.uint8)
    noise = np.random.normal(0, random.uniform(0, 4), canvas.shape)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _draw_distractors(canvas: np.ndarray, count: int) -> None:
    h, w = canvas.shape[:2]
    for _ in range(count):
        color = tuple(int(c) for c in (
            random.randint(40, 180),
            random.randint(40, 180),
            random.randint(40, 180),
        ))
        kind = random.choice(["rect", "circle", "ellipse", "line", "text"])
        thick = random.choice([-1, 1, 2])

        if kind == "rect":
            x1, y1 = random.randint(0, w - 10), random.randint(0, h - 10)
            x2 = random.randint(x1 + 5, min(x1 + 90, w))
            y2 = random.randint(y1 + 5, min(y1 + 90, h))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

        elif kind == "circle":
            cx, cy = random.randint(10, w - 10), random.randint(10, h - 10)
            cv2.circle(canvas, (cx, cy), random.randint(5, 45), color, thick)

        elif kind == "ellipse":
            cx, cy = random.randint(0, w), random.randint(0, h)
            axes = (random.randint(5, 45), random.randint(5, 45))
            cv2.ellipse(canvas, (cx, cy), axes, random.randint(0, 180), 0, 360, color, thick)

        elif kind == "line":
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            cv2.line(canvas, p1, p2, color, random.randint(1, 3))

        elif kind == "text":
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+/"
            text = "".join(random.choices(chars, k=random.randint(1, 6)))
            x, y = random.randint(0, max(1, w - 60)), random.randint(15, h)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        random.uniform(0.3, 1.2), color, random.randint(1, 2), cv2.LINE_AA)


def _apply_image_effects(img: np.ndarray,
                         apply_tint: bool = True,
                         apply_blur: bool = True) -> np.ndarray:
    f = img.copy().astype(np.float32)

    if random.random() < 0.55:
        f += np.random.normal(0, random.uniform(2, 18), f.shape)

    if random.random() < 0.5:
        f *= random.uniform(0.72, 1.28)

    if random.random() < 0.4:
        mean = f.mean()
        f = (f - mean) * random.uniform(0.8, 1.25) + mean

    if apply_tint and random.random() < 0.6:
        if random.random() < 0.55:  # warm / incandescent
            f[:, :, 0] *= random.uniform(0.78, 0.95)
            f[:, :, 2] *= random.uniform(1.0, 1.18)
        else:                        # cool / fluorescent
            f[:, :, 0] *= random.uniform(1.0, 1.18)
            f[:, :, 2] *= random.uniform(0.78, 0.95)

    # Low-light vignette path
    if random.random() < 0.25:
        f *= random.uniform(0.55, 0.85)
        f += np.random.normal(0, random.uniform(8, 22), f.shape)
        rows, cols = f.shape[:2]
        kx = cv2.getGaussianKernel(cols, cols / 2)
        ky = cv2.getGaussianKernel(rows, rows / 2)
        mask = (ky * kx.T)
        mask = (mask / mask.max())[:, :, np.newaxis]
        f *= mask

    result = np.clip(f, 0, 255).astype(np.uint8)

    if apply_blur and random.random() < 0.3:
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SymbolDatasetGenerator:
    def __init__(self, symbols_dir: str, output_dir: str = "dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.symbol_classes = _load_symbols(symbols_dir)
        self._cat_to_id = {c["name"]: c["id"] for c in SYMBOL_CATEGORIES}
        self._next_cat_id = len(SYMBOL_CATEGORIES) + 1

        print(f"Loaded {len(self.symbol_classes)} symbol classes:")
        for name, variants in self.symbol_classes.items():
            print(f"  {name}: {len(variants)} variant(s)")

    def generate(
        self,
        num_images: int = 200,
        img_width: int = 640,
        img_height: int = 640,
        symbols_per_image: Tuple[int, int] = (3, 14),
        symbol_min_size: int = 12,
        symbol_max_size: int = 100,
        rotations: Optional[List[float]] = None,
        symbol_overlap_pad: int = 4,
        distractors_per_image: Tuple[int, int] = (0, 0),
        apply_symbol_effects: bool = True,
        apply_image_effects: bool = True,
        apply_tint: bool = True,
        apply_blur: bool = True,
    ) -> Dict:
        if rotations is None:
            rotations = [0.0, 90.0, 180.0, 270.0]

        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        cat_to_id = dict(self._cat_to_id)
        next_cat_id = self._next_cat_id
        all_classes = list(self.symbol_classes.keys())

        coco: Dict = {
            "images": [],
            "annotations": [],
            "categories": [dict(c) for c in SYMBOL_CATEGORIES],
        }
        ann_id = 1

        for i in range(num_images):
            image_id = i + 1
            canvas = _make_background(img_width, img_height)

            num_distractors = random.randint(*distractors_per_image)
            if num_distractors > 0:
                _draw_distractors(canvas, num_distractors)

            placed_boxes: List[Tuple[int, int, int, int]] = []
            image_anns = []
            num_symbols = random.randint(*symbols_per_image)
            budget = num_symbols * 25

            while len(image_anns) < num_symbols and budget > 0:
                budget -= 1
                cls = random.choice(all_classes)
                src = random.choice(self.symbol_classes[cls])

                target_px = random.randint(symbol_min_size, symbol_max_size)
                sym = _resize_to_target(src.copy(), target_px)

                if apply_symbol_effects:
                    sym = _apply_symbol_effects(sym)

                angle = random.choice(rotations) if rotations else random.uniform(0, 360)
                sym = _rotate(sym, angle)

                sh, sw = sym.shape[:2]
                if sw > img_width or sh > img_height:
                    continue

                x = random.randint(0, img_width - sw)
                y = random.randint(0, img_height - sh)

                if _overlaps(x, y, sw, sh, placed_boxes, symbol_overlap_pad):
                    continue

                _blend(canvas, sym, x, y)
                placed_boxes.append((x, y, sw, sh))

                if cls not in cat_to_id:
                    cat_to_id[cls] = next_cat_id
                    coco["categories"].append({"id": next_cat_id, "name": cls})
                    next_cat_id += 1

                image_anns.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_to_id[cls],
                    "bbox": [float(x), float(y), float(sw), float(sh)],
                    "area": float(sw * sh),
                    "iscrowd": 0,
                })
                ann_id += 1

            if apply_image_effects:
                canvas = _apply_image_effects(canvas, apply_tint=apply_tint, apply_blur=apply_blur)

            fname = f"symbol_{i:04d}.png"
            cv2.imwrite(str(self.images_dir / fname), canvas)

            coco["images"].append({
                "id": image_id,
                "file_name": fname,
                "width": img_width,
                "height": img_height,
            })
            coco["annotations"].extend(image_anns)

            print(f"[{i+1}/{num_images}] {fname} — {len(image_anns)} symbols, {num_distractors} distractors")

        ann_path = self.output_dir / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"\nDone. {num_images} images → '{self.output_dir}/'")
        print(f"  Total annotations: {len(coco['annotations'])}")
        return coco


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate symbol-focused training dataset")
    parser.add_argument("--symbols-dir", type=str, default="data/electrical-symbols")
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=640)
    parser.add_argument("--symbols-min", type=int, default=3,
                        help="Min symbols per image")
    parser.add_argument("--symbols-max", type=int, default=14,
                        help="Max symbols per image")
    parser.add_argument("--size-min", type=int, default=12,
                        help="Min symbol output size in px (longer side)")
    parser.add_argument("--size-max", type=int, default=100,
                        help="Max symbol output size in px (longer side)")
    parser.add_argument("--rotations", type=str, default="0,90,180,270",
                        help="Comma-separated angles. Pass 'free' for continuous 0-360.")
    parser.add_argument("--overlap-pad", type=int, default=4,
                        help="Min pixel gap between placed symbols")
    parser.add_argument("--distractors-min", type=int, default=0)
    parser.add_argument("--distractors-max", type=int, default=None,
                        help="Defaults to --distractors-min when omitted")
    parser.add_argument("--no-symbol-effects", action="store_true")
    parser.add_argument("--no-image-effects", action="store_true")
    parser.add_argument("--no-tint", action="store_true")
    parser.add_argument("--no-blur", action="store_true")

    args = parser.parse_args()

    if args.distractors_max is None:
        args.distractors_max = args.distractors_min

    rotations = None if args.rotations == "free" else [
        float(r) for r in args.rotations.split(",")
    ]

    gen = SymbolDatasetGenerator(
        symbols_dir=args.symbols_dir,
        output_dir=args.output_dir,
    )
    gen.generate(
        num_images=args.num_images,
        img_width=args.img_width,
        img_height=args.img_height,
        symbols_per_image=(args.symbols_min, args.symbols_max),
        symbol_min_size=args.size_min,
        symbol_max_size=args.size_max,
        rotations=rotations,
        symbol_overlap_pad=args.overlap_pad,
        distractors_per_image=(args.distractors_min, args.distractors_max),
        apply_symbol_effects=not args.no_symbol_effects,
        apply_image_effects=not args.no_image_effects,
        apply_tint=not args.no_tint,
        apply_blur=not args.no_blur,
    )


if __name__ == "__main__":
    main()
