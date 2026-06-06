"""Image selection and optimal observed<->predicted matching.

Shared by every model; extracted from the per-model ``compute_model`` bodies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def select_images(images, n_obs: int):
    """Pick the ``n_obs`` predicted images that should match the observations.

    Mirrors the legacy rule: if exactly ``n_obs + 1`` images are found (e.g. a
    faint central image from cuspy profiles), drop the lowest-magnification one;
    if the count is anything other than ``n_obs`` afterwards, return ``None``
    (the candidate is rejected by the objective).
    """
    images = list(images)
    if len(images) == n_obs + 1:
        abs_mags = [abs(im[2]) for im in images]
        drop = int(np.argmin(abs_mags))
        images = [im for i, im in enumerate(images) if i != drop]
    if len(images) != n_obs:
        return None
    return images


def match_images(obs_positions: np.ndarray,
                 pred_positions: np.ndarray,
                 pred_mags: np.ndarray,
                 center_offset: tuple[float, float] = (0.0, 0.0)):
    """Hungarian-match predicted to observed images.

    Returns ``(matched_pos, matched_mag, delta_pos_mas)`` where ``delta_pos_mas``
    is the per-image separation in milliarcsec.
    """
    pred = np.array(pred_positions, dtype=float).copy()
    pred[:, 0] += center_offset[0]
    pred[:, 1] += center_offset[1]

    distances = cdist(obs_positions, pred)
    row_ind, col_ind = linear_sum_assignment(distances)
    order = col_ind[np.argsort(row_ind)]

    matched_pos = pred[order]
    matched_mag = np.asarray(pred_mags, dtype=float)[order]
    delta_pos_mas = np.sqrt(
        np.sum(((matched_pos - obs_positions) * 1000.0) ** 2, axis=1))
    return matched_pos, matched_mag, delta_pos_mas
