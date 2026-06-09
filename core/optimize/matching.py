"""Image selection and optimal observed<->predicted matching.

Shared by every model; extracted from the per-model ``compute_model`` bodies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def select_images(images, n_obs: int, allow_partial: bool = False):
    """Pick the predicted images that should match the observations.

    Mirrors the legacy rule: if exactly ``n_obs + 1`` images are found (e.g. a
    faint central image from cuspy profiles), drop the lowest-magnification one;
    if the count is anything other than ``n_obs`` afterwards, return ``None``
    (the candidate is rejected by the objective).

    With ``allow_partial`` (used when ``missing_img_penalty`` is active) a result
    with *fewer* than ``n_obs`` images is no longer rejected: the available
    images (possibly an empty list) are returned so the caller can score them and
    add a per-missing-image penalty. An *over*-imaged result (more than
    ``n_obs + 1``) is still rejected either way.
    """
    images = list(images)
    if len(images) == n_obs + 1:
        abs_mags = [abs(im[2]) for im in images]
        drop = int(np.argmin(abs_mags))
        images = [im for i, im in enumerate(images) if i != drop]
    if len(images) == n_obs:
        return images
    if allow_partial and len(images) < n_obs:
        return images
    return None


def assign_images(obs_positions: np.ndarray,
                  pred_positions: np.ndarray,
                  pred_mags: np.ndarray,
                  center_offset: tuple[float, float] = (0.0, 0.0)):
    """Hungarian-match predicted to observed images, supporting ``n_pred <= n_obs``.

    Returns ``(matched_pos, matched_mag, delta_pos_mas, obs_idx)`` where
    ``obs_idx`` are the (ascending) observed indices that got matched -- length
    ``n_pred``. For the square case (``n_pred == n_obs``) ``obs_idx`` is
    ``0..n_obs-1`` and the result is identical to :func:`match_images`.
    """
    pred = np.array(pred_positions, dtype=float).copy()
    pred[:, 0] += center_offset[0]
    pred[:, 1] += center_offset[1]

    distances = cdist(obs_positions, pred)              # (n_obs, n_pred)
    row_ind, col_ind = linear_sum_assignment(distances)  # len == min(n_obs, n_pred)
    order = np.argsort(row_ind)
    obs_idx = row_ind[order]
    pred_idx = col_ind[order]

    matched_pos = pred[pred_idx]
    matched_mag = np.asarray(pred_mags, dtype=float)[pred_idx]
    delta_pos_mas = np.sqrt(
        np.sum(((matched_pos - obs_positions[obs_idx]) * 1000.0) ** 2, axis=1))
    return matched_pos, matched_mag, delta_pos_mas, obs_idx


def match_images(obs_positions: np.ndarray,
                 pred_positions: np.ndarray,
                 pred_mags: np.ndarray,
                 center_offset: tuple[float, float] = (0.0, 0.0)):
    """Hungarian-match predicted to observed images.

    Returns ``(matched_pos, matched_mag, delta_pos_mas)`` where ``delta_pos_mas``
    is the per-image separation in milliarcsec. Assumes ``n_pred == n_obs`` (the
    caller has already run :func:`select_images`); for the partial case use
    :func:`assign_images`, which this wraps.
    """
    matched_pos, matched_mag, delta_pos_mas, _ = assign_images(
        obs_positions, pred_positions, pred_mags, center_offset)
    return matched_pos, matched_mag, delta_pos_mas
