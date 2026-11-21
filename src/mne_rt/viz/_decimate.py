"""Peak decimation: draw a dense trace with the points a screen can show."""

from __future__ import annotations

import numpy as np


def peak_decimate(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Each of ``n_bins`` columns' minimum and maximum, interleaved.

    A pixel column of a plot can show no more than the span from its lowest to
    its highest sample, so ``2 * n_bins`` points draw what ``len(x)`` do: spikes
    and artifacts keep their full height. The bins differ in size by at most
    one sample, so every sample is in one, the first and the newest included.

    Parameters
    ----------
    x : ndarray, shape (n_times,)
        Sample positions, increasing.
    y : ndarray, shape (n_times,) or (n_traces, n_times)
        One trace, or several sharing ``x``.
    n_bins : int
        Columns to reduce to, typically the plot's width in device pixels.

    Returns
    -------
    x, y : ndarray
        ``x`` of shape ``(2 * n_bins,)`` and ``y`` shaped like the input with
        ``2 * n_bins`` samples, or the inputs unchanged when they already have
        no more than ``2 * n_bins`` samples.
    """
    n = x.shape[0]
    if n_bins < 1 or n <= 2 * n_bins:
        return x, y
    # Equal bins would leave out the n % n_bins samples that do not fill one:
    # up to a third of a window, drawn as a blank left edge.
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    starts, ends = edges[:-1], edges[1:] - 1
    out = np.empty(y.shape[:-1] + (n_bins, 2), dtype=y.dtype)
    out[..., 0] = np.minimum.reduceat(y, starts, axis=-1)
    out[..., 1] = np.maximum.reduceat(y, starts, axis=-1)
    x_out = np.empty(2 * n_bins, dtype=x.dtype)
    x_out[0::2] = x[starts]
    x_out[1::2] = x[ends]
    return x_out, out.reshape(y.shape[:-1] + (2 * n_bins,))
