from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple
import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2

from utils.inspection import inspect_folder


def generate_colors(num_colors: int) -> np.ndarray:
    """Generate an array with RGB triplets representing colors."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, size=(num_colors, 3))


_DEFAULT_COLORS = generate_colors(1001)


def show_plain_image(image_path: str) -> Optional[np.ndarray]:
    """
    Display the image and print basic metadata.
    Converts BGR to RGB for correct matplotlib visualization and handles missing files.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    print("-" * 70)
    print(f"Image Shape: {img.shape}")
    print(f"Data Type: {img.dtype}")
    print(f"Pixel values range: Min={img.min()}, Max={img.max()}")
    print("-" * 70)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Preview: {os.path.basename(image_path)}")
    plt.axis("off")
    plt.show()
    return img


def get_prw_data(dataset_root: str, index: int, folder_img: str = "frames"):
    """
    Load the image and its corresponding annotations for a given index.

    Returns:
        image (PIL.Image): loaded image.
        boxes_xyxy (List): bounding boxes in [xmin, ymin, xmax, ymax] format.
        ids (List): person IDs (int).
    """
    img_path = inspect_folder(dataset_root, folder=folder_img, index=index)
    if img_path is None:
        return None, [], []

    image = Image.open(img_path).convert("RGB")
    if folder_img == "query_box":
        return image, [], []

    mat_path = inspect_folder(dataset_root, folder="annotations", index=index)
    if mat_path is None or not os.path.exists(mat_path):
        return image, [], []

    boxes_xyxy = []
    ids = []

    mat_data = scipy.io.loadmat(mat_path)
    valid_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    raw_data = mat_data[valid_keys[0]] if valid_keys else []

    for row in raw_data:
        pid, x, y, w, h = row
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        boxes_xyxy.append([x_min, y_min, x_max, y_max])
        ids.append(int(pid))

    return image, boxes_xyxy, ids


def draw_boxes(image: Image.Image, boxes, labels, colors, add_text: bool = True) -> Image.Image:
    """Draw rectangles and ID labels on the image."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    image_with_bb = copy.deepcopy(image)
    painter = ImageDraw.Draw(image_with_bb)

    for box, label in zip(boxes, labels):
        color_idx = label if label >= 0 else len(colors) - 1
        color = tuple(colors[color_idx % len(colors)].astype(np.int32))

        painter.rectangle(box, outline=color, width=3)

        if add_text:
            text_in_box = f"ID: {label}" if label != -2 else "Unknown"
            left, top, right, bottom = font.getbbox(text_in_box)
            text_width = right - left
            text_height = bottom - top
            x_min, y_min = box[0], box[1]

            painter.rectangle(
                [(x_min, y_min - text_height - 4), (x_min + text_width + 4, y_min)],
                fill=color,
            )
            painter.text(
                (x_min + 2, y_min - text_height - 4),
                text_in_box,
                fill="black",
                font=font,
            )

    return image_with_bb


def render_prw_frame(
    dataset_root: str,
    sample_index: int,
    files: Sequence[str],
    show_annotations: bool = True,
    folder_img: str = "frames",
    colors: Optional[np.ndarray] = None,
) -> None:
    """Handle PRW image rendering logic with optional annotations."""
    image, boxes, ids = get_prw_data(dataset_root, sample_index, folder_img)
    if image is None:
        return

    file_name = files[sample_index]
    if not show_annotations:
        boxes = []
        ids = []

    if boxes:
        display_img = draw_boxes(
            image,
            boxes=boxes,
            labels=ids,
            colors=colors if colors is not None else _DEFAULT_COLORS,
            add_text=True,
        )
        title = f"File: {file_name} - Persons Detected: {len(boxes)}"
    else:
        display_img = image
        title = f"File: {file_name} - (No annotations found)" if show_annotations else f"File: {file_name} - (Raw Image)"

    plt.figure(figsize=(12, 8))
    plt.imshow(display_img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def analyze_image_shapes(folder_path: str):
    """Scan a folder and count the distribution of image resolutions."""
    shapes_count = {}

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return {}

    print(f"Analyzing images in: {folder_path}")
    files = sorted(os.listdir(folder_path))

    for filename in files:
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        resolution = f"{w}x{h}"
        shapes_count[resolution] = shapes_count.get(resolution, 0) + 1

    print("\nResolution Distribution:")
    for res, count in shapes_count.items():
        print(f"  - {res}: {count} images")

    return shapes_count


def tensor_to_pil(tensor_img) -> Image.Image:
    """Convert a PyTorch float32 Tensor [0,1] to a PIL Image."""
    tensor = tensor_img.clone().clamp(0, 1)
    return v2.ToPILImage()(tensor)


def render_dataset_sample(dataset, index: int, show_annotations: bool = True, colors: Optional[np.ndarray] = None) -> None:
    """Render a dataset sample with optional bounding boxes."""
    img_tensor, target = dataset[index]
    img_pil = tensor_to_pil(img_tensor)

    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()
    img_name = target.get("img_name", str(index))

    if not show_annotations:
        boxes = []
        labels = []

    if boxes:
        display_img = draw_boxes(
            img_pil,
            boxes=boxes,
            labels=labels,
            colors=colors if colors is not None else _DEFAULT_COLORS,
            add_text=True,
        )
        title = f"Index: {index} - Image: {img_name} - Persons Detected: {len(boxes)}"
    else:
        display_img = img_pil
        title = f"Index: {index} - Image: {img_name} - (Clean Image)"

    plt.figure(figsize=(12, 8))
    plt.imshow(display_img)
    plt.title(title)
    plt.axis("off")
    plt.show()
