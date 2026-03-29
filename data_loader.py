"""
data_loader.py
Discovers all labelled images from BioID and LFW.

BioID layout  (flat directory of .pgm files, one identity per file):
    <root>/BioID_0000.pgm  ...

    BioID has 23 subjects × ~20 images each.
    The subject ID is encoded in the filename: BioID_XYYY.pgm
      X   = subject index (0-based, shifts at image 0093, 0188 …)
    We derive subject IDs from the official BioID index ranges.

LFW layout (nested, one folder per person):
    <root>/lfw_funneled/<Person_Name>/<Person_Name>_0001.jpg  ...
    or
    <root>/<Person_Name>/<Person_Name>_0001.jpg  ...

    We keep only persons with >= MIN_SAMPLES images so the classifier
    has enough examples per class.
"""

from pathlib import Path
import re

# Only keep LFW identities that have at least this many images
LFW_MIN_SAMPLES = 10


# BioID 

# Official BioID subject boundaries
# Each subject contributes ~20 consecutive images
_BIOID_BOUNDARIES = [
    0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460,
]

def _bioid_label(image_index: int) -> int:
    """Map a BioID image index to a subject ID (0-based)."""
    for i, boundary in enumerate(reversed(_BIOID_BOUNDARIES)):
        if image_index >= boundary:
            return len(_BIOID_BOUNDARIES) - 1 - i
    return 0


def load_bioid(data_dir: str) -> tuple[list[Path], list[int]]:
    """
    Returns (image_paths, labels) for all BioID .pgm files found.
    """
    root   = Path(data_dir)
    images = sorted(root.glob("*.pgm"))
    if not images:
        images = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(f"No BioID images found in: {data_dir}")

    paths, labels = [], []
    for p in images:
        m = re.search(r"(\d+)", p.stem)
        idx = int(m.group(1)) if m else 0
        paths.append(p)
        labels.append(_bioid_label(idx))

    return paths, labels


# LFW 

def load_lfw(data_dir: str, min_samples: int = LFW_MIN_SAMPLES) -> tuple[list[Path], list[int], list[str]]:
    """
    Returns (image_paths, labels, label_names).
    Only includes identities with >= min_samples images.
    """
    root = Path(data_dir)

    person_dirs = [d for d in root.rglob("*") if d.is_dir()]
    person_dirs = [d for d in person_dirs if not d.name.startswith(".")]

    label_map: dict[str, list[Path]] = {}
    for d in person_dirs:
        imgs = (
            list(d.glob("*.jpg"))
            + list(d.glob("*.jpeg"))
            + list(d.glob("*.png"))
        )
        if len(imgs) >= min_samples:
            label_map[d.name] = imgs

    if not label_map:
        raise FileNotFoundError(
            f"No LFW person directories with >= {min_samples} images found in: {data_dir}\n"
            f"Make sure data_dir points to the folder that contains per-person sub-folders."
        )

    label_names = sorted(label_map.keys())
    paths, labels = [], []
    for label_id, name in enumerate(label_names):
        for img_path in label_map[name]:
            paths.append(img_path)
            labels.append(label_id)

    return paths, labels, label_names
