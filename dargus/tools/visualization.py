"""Visualization helpers."""

from __future__ import annotations

import io

import numpy as np


def save_figure_bytes(fig) -> bytes:
    """Save a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def make_volcano_plot(
    logfc: np.ndarray,
    pvalues: np.ndarray,
    title: str = "Volcano plot",
) -> bytes:
    """Create a volcano plot and return PNG bytes."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib unavailable") from exc

    fig, ax = plt.subplots(figsize=(6, 5))
    logp = -np.log10(np.clip(pvalues, 1e-300, 1))
    ax.scatter(logfc, logp, alpha=0.5, s=10)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--")
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 p-value")
    ax.set_title(title)
    return save_figure_bytes(fig)
