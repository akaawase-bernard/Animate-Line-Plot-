#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 13:19:51 2026

@author: bernard
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def make_timeseries_gif(
    y,
    x=None,
    filename="timeseries.gif",
    fps=30,
    duration=5,
    show_cursor=True,
    dpi=150,
    line_kwargs=None
):
    """
    Create a left-to-right revealing GIF for a time series.
    """

    # ---------------------------
    # Input validation
    # ---------------------------
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D")
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

    if not np.all(np.diff(x) >= 0):
        raise ValueError("x must be monotonically increasing")

    n_frames = max(2, int(fps * duration))
    line_kwargs = line_kwargs or {}

    # ---------------------------
    # Ensure output directory exists
    # ---------------------------
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # ---------------------------
    # Figure setup
    # ---------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.set_xlim(x.min(), x.max())

    y_min, y_max = np.nanmin(y), np.nanmax(y)
    pad = 0.05 * (y_max - y_min if y_max != y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    (line,) = ax.plot([], [], **line_kwargs)

    cursor = None
    if show_cursor:
        cursor = ax.axvline(x=x.min(), color="k", lw=1, alpha=0.6)

    # ---------------------------
    # Animation logic
    # ---------------------------
    def update(frame):
        frac = frame / (n_frames - 1)
        x_end = x.min() + frac * (x.max() - x.min())

        mask = x <= x_end
        line.set_data(x[mask], y[mask])

        if cursor is not None:
            cursor.set_xdata([x_end, x_end])  # FIXED

        return (line,) if cursor is None else (line, cursor)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False  # Spyder-safe
    )

    # ---------------------------
    # Save GIF
    # ---------------------------
    anim.save(
        filename,
        writer=PillowWriter(fps=fps),
        dpi=dpi
    )

    plt.close(fig)


# ===========================
# Examples
# ===========================

y = np.cumsum(np.random.randn(300))
make_timeseries_gif(y, filename="data/random_walk.gif")

t = np.linspace(0, 10, 500)
y = np.sin(t) * np.exp(-0.1 * t)

make_timeseries_gif(
    y,
    x=t,
    filename="data/damped_wave.gif",
    line_kwargs=dict(color="royalblue", lw=2)
)
