
import numpy as np


def matrix_image(ax, values, *, cmap, norm):
    """Keep every matrix value distinct in a single nearest-neighbor image."""
    rows, columns = values.shape
    return ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=(0, columns, rows, 0),
        interpolation="none",
        resample=False,
        aspect="equal",
    )


def continuous_colorbar(figure, mappable, *, cax, orientation="vertical"):
    """Draw one gradient image, retaining standard colorbar ticks and labels."""
    colorbar = figure.colorbar(mappable, cax=cax, orientation=orientation)
    colorbar.solids.set_visible(False)
    x0, x1 = cax.get_xlim()
    y0, y1 = cax.get_ylim()
    values = np.linspace(mappable.norm.vmin, mappable.norm.vmax, 1024)
    gradient = values[None, :] if orientation == "horizontal" else values[:, None]
    cax.imshow(
        gradient,
        cmap=mappable.cmap,
        norm=mappable.norm,
        origin="lower",
        extent=(x0, x1, y0, y1),
        aspect="auto",
        interpolation="none",
        resample=False,
    )
    colorbar.outline.set_visible(False)
    cax.tick_params(width=0.5, length=2, colors="#333333")
    return colorbar
