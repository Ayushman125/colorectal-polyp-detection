from __future__ import annotations

import numpy as np


def dense_crf_refine(
    image_rgb: np.ndarray,
    foreground_probability: np.ndarray,
    iterations: int = 5,
) -> np.ndarray:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be a uint8 RGB image")
    if foreground_probability.shape != image_rgb.shape[:2]:
        raise ValueError("Probability map and image dimensions must match")
    if not np.isfinite(foreground_probability).all():
        raise ValueError("Probability map contains non-finite values")
    if np.any(foreground_probability < 0) or np.any(foreground_probability > 1):
        raise ValueError("Foreground probabilities must be in [0, 1]")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    try:
        import pydensecrf.densecrf as densecrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError as error:
        raise RuntimeError(
            "Dense CRF evaluation requires pydensecrf. Install the optional "
            "dependency using the instructions in README.md."
        ) from error

    height, width = foreground_probability.shape
    probabilities = np.stack(
        (1.0 - foreground_probability, foreground_probability)
    ).astype(np.float32)
    probabilities = np.clip(probabilities, 1e-6, 1.0)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    crf = densecrf.DenseCRF2D(width, height, 2)
    crf.setUnaryEnergy(unary_from_softmax(probabilities))
    crf.addPairwiseGaussian(sxy=3, compat=3)
    crf.addPairwiseBilateral(
        sxy=50,
        srgb=13,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=10,
    )
    refined = np.asarray(crf.inference(iterations), dtype=np.float32)[1]
    return refined.reshape(height, width)
