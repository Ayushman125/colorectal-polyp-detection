from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_patch_features(
    image: np.ndarray, grid_size: int = 4, color_bins: int = 8
) -> np.ndarray:
    """Extract image-only handcrafted descriptors from a regular image grid.

    The ground-truth mask is deliberately not an input: patch labels are used
    only by the training routine, avoiding target leakage into descriptors.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB image with shape (H, W, 3)")
    height, width = image.shape[:2]
    descriptors: list[list[float]] = []
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for row in range(grid_size):
        y0, y1 = row * height // grid_size, (row + 1) * height // grid_size
        for col in range(grid_size):
            x0, x1 = col * width // grid_size, (col + 1) * width // grid_size
            rgb_patch = image[y0:y1, x0:x1]
            hsv_patch = hsv[y0:y1, x0:x1]
            lab_patch = lab[y0:y1, x0:x1]
            gray_patch = gray[y0:y1, x0:x1]
            if min(gray_patch.shape) < 4:
                raise ValueError("grid_size creates patches smaller than 4 pixels")

            values: list[float] = []
            for color_patch, maximum in (
                (hsv_patch, (180, 256, 256)),
                (lab_patch, (256, 256, 256)),
            ):
                for channel in range(3):
                    plane = color_patch[..., channel]
                    values.extend(
                        float(v)
                        for v in cv2.calcHist(
                            [plane],
                            [0],
                            None,
                            [color_bins],
                            [0, maximum[channel]],
                        ).flatten()
                        / plane.size
                    )
                    values.extend(
                        (float(np.mean(plane)) / 255.0, float(np.std(plane)) / 255.0)
                    )

            quantized = np.clip(gray_patch.astype(np.uint16) * 16 // 256, 0, 15).astype(
                np.uint8
            )
            glcm = graycomatrix(
                quantized,
                distances=[1, 3],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=16,
                symmetric=True,
                normed=True,
            )
            for property_name in ("contrast", "homogeneity", "energy", "correlation"):
                values.extend(
                    graycoprops(glcm, property_name).astype(np.float32).ravel().tolist()
                )

            lbp = local_binary_pattern(gray_patch, P=8, R=1, method="uniform")
            lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))
            values.extend((lbp_hist / max(1, lbp_hist.sum())).tolist())

            edges = cv2.Canny(gray_patch, 60, 160)
            edge_fraction = float(np.mean(edges > 0))
            gx = cv2.Sobel(gray_patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_patch, cv2.CV_32F, 0, 1, ksize=3)
            gradient = cv2.magnitude(gx, gy)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(contour))
                perimeter = float(cv2.arcLength(contour, True))
                _, _, box_width, box_height = cv2.boundingRect(contour)
                circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-8)
                aspect_ratio = box_width / max(1, box_height)
                area_fraction = area / (gray_patch.shape[0] * gray_patch.shape[1])
            else:
                circularity, aspect_ratio, area_fraction = 0.0, 0.0, 0.0
            values.extend(
                (
                    edge_fraction,
                    float(np.mean(gradient)) / 1020.0,
                    float(np.std(gradient)) / 1020.0,
                    float(circularity),
                    float(aspect_ratio),
                    float(area_fraction),
                )
            )
            descriptors.append(values)
    return np.asarray(descriptors, dtype=np.float32)


def label_patches(mask: np.ndarray, grid_size: int = 4, positive_fraction: float = 0.02) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    labels: list[int] = []
    height, width = mask.shape
    for row in range(grid_size):
        y0, y1 = row * height // grid_size, (row + 1) * height // grid_size
        for col in range(grid_size):
            x0, x1 = col * width // grid_size, (col + 1) * width // grid_size
            labels.append(int(float(np.mean(mask[y0:y1, x0:x1] > 0)) >= positive_fraction))
    return np.asarray(labels, dtype=np.uint8)
