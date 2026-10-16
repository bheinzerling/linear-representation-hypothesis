"""Figure 3(a,b) renderers derived from the existing figure on 2026-09-09."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
BLUE = COSINE = "#35618D"
GREEN = SINE = "#3F7F64"
OCHRE = "#AC7D00"
LIGHT_GREEN = "#E7F1EA"
DARK, MID, LIGHT = "#25231F", "#73706A", "#D8D3CA"
PAIR = 3
CENTER = GAIN = 499.5
CIRCULAR_ARROW_INCHES = .32
elevation, azimuth = np.radians([24, -110])
base = np.array([[-np.sin(azimuth), np.cos(azimuth), 0],
                 [-np.sin(elevation)*np.cos(azimuth), -np.sin(elevation)*np.sin(azimuth), np.cos(elevation)]])
roll = -np.arctan2(base[1, 0], base[0, 0])
CAMERA = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]]) @ base


def prepare_axis(ax: plt.Axes, title: str, half_width: float = 1.05) -> None:
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-0.76, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", pad=7, fontweight="bold")

def arrow(
    ax: plt.Axes,
    start: tuple[float, float] | np.ndarray,
    end: tuple[float, float] | np.ndarray,
    color: str = MID,
    style: str = "-|>",
    linewidth: float = 1.2,
    linestyle: str | tuple = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle=style,
            mutation_scale=7,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            zorder=3,
        )
    )

def draw_local_linearization(ax: plt.Axes) -> None:
    prepare_axis(ax, "(a) Local linearization")
    ax.set_ylim(-1.0, 1.05)
    x = np.linspace(-0.91, 0.91, 200)
    y = -0.01 + 0.18 * x + 0.50 * x**2
    x0 = 0.08
    y0 = -0.01 + 0.18 * x0 + 0.50 * x0**2
    slope = 0.18 + x0

    ax.axvspan(x0 - 0.24, x0 + 0.24, ymin=0.24, ymax=0.91, color=LIGHT_GREEN)
    ax.plot(x, y, color=BLUE, linewidth=2.0, zorder=2)
    tangent_x = np.array([-0.84, 0.84])
    ax.plot(
        tangent_x,
        y0 + slope * (tangent_x - x0),
        color=OCHRE,
        linewidth=1.4,
        linestyle=(0, (4, 2)),
        zorder=3,
    )
    local_x = np.array([x0 - 0.24, x0 + 0.24])
    ax.plot(local_x, y0 + slope * (local_x - x0), color=OCHRE, linewidth=1.8)
    ax.scatter([x0], [y0], s=24, color=DARK, edgecolor="white", linewidth=0.5, zorder=4)
    ax.plot([x0, x0], [-0.52, y0], color=LIGHT, linewidth=0.7, linestyle=(0, (2, 2)))
    arrow(ax, (-0.95, -0.53), (0.98, -0.53), linewidth=0.65)
    ax.text(x0, -0.58, r"$h_0$", ha="center", va="top")
    ax.text(0.0, -0.98, "activation (1D slice)", ha="center", va="bottom", color=MID)
    ax.text(-0.91, 0.59, "smooth decoder", color=BLUE, ha="left", va="center")
    ax.text(0.35, -0.13, "affine\napproximation", color=OCHRE, ha="center", va="top")
    ax.text(x0, 0.98, "evaluation\nregion", color=GREEN, ha="center", va="center")

def draw_probe_outputs(ax: plt.Axes, frame: pd.DataFrame, source: dict, *, footer_offset: float = 0) -> None:
    ax.set_axis_off()
    inset = [.015, .18, .97, .80]
    plot = ax.inset_axes(inset)
    plot.set_axis_off()
    plot.set_aspect("equal")
    parent = ax.get_position()
    viewport = np.array([
        ax.figure.get_figwidth() * parent.width * inset[2],
        ax.figure.get_figheight() * parent.height * inset[3],
    ])
    plot.set_xlim(0, viewport[0])
    plot.set_ylim(0, viewport[1])
    values = frame.value.to_numpy()
    points = np.column_stack([
        (frame.ridge_prediction.to_numpy() - CENTER) / GAIN,
        frame[f"circular_cos_{PAIR}"], frame[f"circular_sin_{PAIR}"],
    ])
    view = points @ CAMERA.T
    omega = source["circular_pairs"][PAIR]["angular_frequency"]
    support = np.linspace(0, 999, 2000)
    reference = np.column_stack([
        (support - CENTER) / GAIN, np.cos(omega * support), np.sin(omega * support),
    ]) @ CAMERA.T
    # As in G, a transverse translation centers the ridge axis in the cloud.
    axis_height = (view[:, 1].min() + view[:, 1].max()) / 2
    transverse = CAMERA[1, 1:]
    anchor = transverse * axis_height / (transverse @ transverse)
    score_values = np.array([0., 500., 999.])
    score_points = np.column_stack([
        (score_values - CENTER) / GAIN, np.tile(anchor, (3, 1)),
    ]) @ CAMERA.T
    bounds = np.vstack((view, reference, score_points))
    midpoint = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    scale = float(np.min((viewport - .09) / np.ptp(bounds, axis=0)))

    def display(xy):
        return (xy - midpoint) * scale + viewport / 2

    projected = display(view)
    plot.plot(*display(reference).T, color=MID, lw=.45, alpha=.65, zorder=1)
    order = np.argsort(points @ np.cross(CAMERA[0], CAMERA[1]), kind="stable")
    scatter = plot.scatter(
        *projected[order].T, c=values[order], cmap="viridis", vmin=0, vmax=999,
        s=.85, alpha=.72, marker="o", edgecolors="none", linewidths=0, zorder=2,
    )
    axis = display(np.array([[bounds[:, 0].min(), axis_height],
                             [bounds[:, 0].max(), axis_height]]))
    plot.add_patch(FancyArrowPatch(
        axis[0], axis[1], arrowstyle="-|>", mutation_scale=6.5,
        color=OCHRE, lw=.9, shrinkA=0, shrinkB=0, zorder=4,
    ))
    ticks = display(score_points)
    for value, tick in zip(score_values, ticks, strict=True):
        plot.plot([tick[0], tick[0]], [tick[1] - .02, tick[1] + .02],
                  color=OCHRE, lw=.55, zorder=5)
        plot.annotate(f"{value:.0f}", xy=tick, xytext=(0, -7), textcoords="offset points",
                      ha="center", va="center", fontsize=7, color=OCHRE, zorder=6)
    plot.annotate(r"$\hat v_{\rm ridge}$", xy=ticks[1], xytext=(0, 7),
                  textcoords="offset points", ha="center", va="bottom",
                  color=OCHRE, fontsize=7.5, zorder=6)
    # The two circular-output axes start exactly at predicted numerical value 0.
    origin = ticks[0]
    arrows = []
    for component, label, color, offset, ha in [
        (1, r"$\hat c$", COSINE, (-.5, -.5), "right"),
        (2, r"$\hat s$", SINE, (.5, -.5), "left"),
    ]:
        end = origin + CIRCULAR_ARROW_INCHES * CAMERA[:, component]
        arrow = FancyArrowPatch(
            origin, end, arrowstyle="-|>", mutation_scale=6,
            color=color, lw=.8, shrinkA=0, shrinkB=0, zorder=7,
        )
        plot.add_patch(arrow)
        plot.annotate(label, xy=end, xytext=offset, textcoords="offset points",
                      ha=ha, va="bottom", color=color, fontsize=8, zorder=8)
        arrows.append((origin.copy(), end.copy()))
    plot.scatter(*origin, s=6, color=OCHRE, edgecolor="white", linewidth=.25, zorder=8)
    # One compact key states the semantics of the circular axes and gray curve.
    ax.legend(handles=[
        Line2D([], [], color=COSINE, lw=.8, label=r"$\hat c$: cosine"),
        Line2D([], [], color=SINE, lw=.8, label=r"$\hat s$: sine"),
        Line2D([], [], color=MID, lw=.6, label="target code"),
    ], loc="lower center", bbox_to_anchor=(.5, -.008 + footer_offset), ncol=3, frameon=False,
        fontsize=6.5, handlelength=.65, handletextpad=.25, columnspacing=.7,
        borderaxespad=0)
    bar = ax.inset_axes([.32, .087 + footer_offset, .44, .026])
    bar.imshow(np.linspace(0, 1, 256)[None, :], cmap="viridis", aspect="auto",
               extent=[0, 999, 0, 1], interpolation="nearest")
    bar.set_axis_off()
    bar.text(-.04, .5, "0", transform=bar.transAxes, ha="right", va="center", color=MID, fontsize=7)
    bar.text(1.04, .5, "999", transform=bar.transAxes, ha="left", va="center", color=MID, fontsize=7)
    ax.text(.23, .100 + footer_offset, "value", transform=ax.transAxes, ha="right", va="center",
            color=MID, fontsize=7)
    # Verify point preservation, calibrated score positions, and arrow placement.
    assert len(scatter.get_offsets()) == 1000 and scatter.get_clim() == (0, 999)
    np.testing.assert_allclose(
        (np.asarray(scatter.get_offsets())[np.argsort(order)] - viewport / 2) / scale + midpoint,
        view, atol=1e-14,
    )
    np.testing.assert_allclose(points[:, 0] * GAIN + CENTER, frame.ridge_prediction, atol=1e-12)
    np.testing.assert_allclose(ticks[:, 1], ticks[0, 1], atol=1e-14)
    for start, end in arrows:
        np.testing.assert_array_equal(start, ticks[0])
        assert np.all(end > 0) and np.all(end < viewport)
    assert np.all(projected > 0) and np.all(projected < viewport)
