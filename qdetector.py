#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "plotly"]
# ///
"""Non-parametric multivariate change point detection (Holland 2011 / Li & Jayaweera 2013).

Run the demo:   uv run qdetector.py            (or: python qdetector.py)
Use the class:  Qdetector(samples).detect()    samples: (dims, n) or (n,)
"""
import argparse

import numpy as np


class Qdetector:
    """Quickest detector based on multivariate spatial signs (rank-like, distribution-free).

    Parameters
    ----------
    data  : array (dims, n) or (n,) - one row per sensor / channel.
    ratio : detection fires when var(Rkn) / mean(Rkn) >= ratio.
    """

    def __init__(self, data, ratio=3.0):
        self.data = np.atleast_2d(np.asarray(data, dtype=float))
        self.ratio = ratio
        self.dim, self.n = self.data.shape
        self.Rkn = None
        self.changePoint = None
        self.detectionFlag = False

    def detect(self):
        Z, n, d = self.data, self.n, self.dim
        # Spatial-sign sum R(Z_i) = sum_j (Z_i - Z_j) / ||Z_i - Z_j||
        diff = Z[:, :, None] - Z[:, None, :]                      # (d, n, n)
        norm = np.linalg.norm(diff, axis=0)                       # (n, n)
        R = np.sum(np.divide(diff, norm, out=np.zeros_like(diff), where=norm > 0), axis=2)  # (d, n)
        # Running mean of R over the first k+1 samples
        k = np.arange(1, n + 1)
        Rbar = np.cumsum(R, axis=1) / k                           # (d, n)
        # Covariance estimate per k is a scalar multiple of R R^T
        RRt = R @ R.T + 1e-6 * np.eye(d)
        scale = (n - k) / ((n - 1.0) * n * k)                     # last point is never a change point
        inv = np.linalg.inv(RRt)
        quad = np.einsum("in,ij,jn->n", Rbar, inv, Rbar)
        self.Rkn = np.where(scale > 0, quad / np.where(scale > 0, scale, 1), 0.0)
        self.changePoint = int(np.argmax(self.Rkn))
        self.detectionFlag = bool(np.var(self.Rkn) / np.mean(self.Rkn) >= self.ratio)
        return self.changePoint if self.detectionFlag else None


def simulate(states=(0, 1), sigma=0.8, sensors=4, seed=0):
    """Piecewise-constant ground truth with random sojourn times, observed by noisy sensors."""
    rng = np.random.default_rng(seed)
    sojourn = rng.integers(1, 300, size=len(states))
    sojourn[-1] = 20
    truth = np.repeat(states, sojourn).astype(float)
    samples = truth + sigma * rng.standard_normal((sensors, truth.size))
    return truth, samples


def plot(truth, samples, det):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    x = np.arange(truth.size)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Hidden state (ground truth)", "Noisy sensor readings",
                                        "Test statistic R<sub>k,n</sub>"))
    fig.add_trace(go.Scatter(x=x, y=truth, mode="lines", name="true state", line_shape="hv",
                             line=dict(color="#52514e", width=2)), row=1, col=1)
    for i, s in enumerate(samples):
        fig.add_trace(go.Scatter(x=x, y=s, mode="lines", name=f"sensor {i + 1}",
                                 line=dict(color=colors[i % len(colors)], width=1.5), opacity=0.85), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=det.Rkn, mode="lines", name="R<sub>k,n</sub>",
                             line=dict(color="#1c5cab", width=2)), row=3, col=1)
    label = f"detected change @ {det.changePoint}" if det.detectionFlag else "no change detected"
    fig.add_vline(x=det.changePoint, line=dict(color="#c0392b", dash="dash", width=2), row="all", col=1)
    fig.add_annotation(x=det.changePoint, y=1, yref="y3 domain", text=label, showarrow=False,
                       xanchor="right", xshift=-6, font=dict(color="#c0392b"))
    truth_cp = int(np.argmax(np.diff(truth) != 0)) + 1 if np.any(np.diff(truth)) else None
    if truth_cp is not None:
        fig.add_vline(x=truth_cp, line=dict(color="#52514e", dash="dot", width=1), row="all", col=1)
        fig.add_annotation(x=truth_cp, y=0, yref="y3 domain", text=f"true change @ {truth_cp}", showarrow=False,
                           xanchor="left", xshift=6, yanchor="bottom", font=dict(color="#52514e"))
    fig.update_layout(title="Multivariate non-parametric change point detection",
                      template="plotly_white", hovermode="x unified", height=750,
                      legend=dict(orientation="h", y=-0.08))
    fig.update_xaxes(title_text="sample #", row=3, col=1)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", default="0,1", help="comma-separated hidden state sequence, e.g. 0,0,1,1,0")
    p.add_argument("--sigma", type=float, default=0.8, help="sensor noise std-dev")
    p.add_argument("--sensors", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="-1 for a fresh random draw")
    p.add_argument("--html", metavar="PATH", help="write interactive plot to this file instead of opening a window")
    a = p.parse_args()

    states = [int(s) for s in a.states.split(",")]
    truth, samples = simulate(states, a.sigma, a.sensors, None if a.seed < 0 else a.seed)
    det = Qdetector(samples)
    cp = det.detect()
    print(f"{truth.size} samples x {a.sensors} sensors -> "
          + (f"change point detected at sample {cp}" if cp is not None else "no change point detected"))

    fig = plot(truth, samples, det)
    if a.html:
        fig.write_html(a.html, include_plotlyjs="cdn")
        print("wrote", a.html)
    else:
        fig.show()


if __name__ == "__main__":
    main()
