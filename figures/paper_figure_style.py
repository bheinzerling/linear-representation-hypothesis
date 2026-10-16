
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.backends.backend_cairo import FigureCanvasCairo, RendererCairo, cairo
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# Matplotlib 3.10's hatch renderer uses pycairo enum names. The pinned cairocffi
# backend exposes the equivalent constants; retain the existing hatch patterns.
if not hasattr(cairo, "Content"):
    cairo.Content = SimpleNamespace(COLOR_ALPHA=cairo.CONTENT_COLOR_ALPHA)
if not hasattr(cairo, "Extend"):
    cairo.Extend = SimpleNamespace(REPEAT=cairo.EXTEND_REPEAT)

FONT_FAMILY = "TeX Gyre Heros"
MATH_FONTSET = "cm"


@lru_cache(maxsize=1)
def register_fonts() -> dict:
    fonts = {}
    for style in ("Regular", "Bold", "Italic", "Bold Italic"):
        path = Path(subprocess.check_output(
            ["fc-match", "-f", "%{file}", f"{FONT_FAMILY}:style={style}"],
            text=True,
        ))
        font_manager.fontManager.addfont(path)
        if font_manager.FontProperties(fname=path).get_name() != FONT_FAMILY:
            raise RuntimeError(f"Install TeX Gyre Heros; fontconfig returned {path}")
        fonts[style] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return fonts


def configure_fonts() -> dict:
    """Set font families while keeping each figure's existing sizes and layout."""
    fonts = register_fonts()
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY],
        "mathtext.fontset": MATH_FONTSET,
    })
    return fonts


def apply_fonts(figure) -> None:
    """Also cover text created by imported panel helpers with explicit fonts."""
    for text in figure.findobj(Text):
        # Monospaced prompts/completions distinguish literal model text.
        if not any("mono" in family.lower() for family in text.get_fontfamily()):
            text.set_fontfamily(FONT_FAMILY)
        text.set_math_fontfamily(MATH_FONTSET)


class FigureCanvas(FigureCanvasCairo):
    """Cairo PDF export at print resolution, including tight-bbox handling."""

    def print_pdf(self, filename, *, metadata=None, orientation="portrait",
                  dpi=None, facecolor=None, edgecolor=None, bbox_inches_restore=None):
        # FigureCanvasBase has already applied the requested DPI and export crop.
        # Cairo's stock PDF canvas resets DPI to 72, which degrades raster artists.
        figure = self.figure
        dpi = figure.dpi
        width, height = figure.get_size_inches() * 72
        if orientation != "portrait":
            raise ValueError("Manuscript figure PDFs use portrait page orientation")
        surface = cairo.PDFSurface(filename, width, height)
        surface.restrict_to_version(cairo.PDF_VERSION_1_5)
        try:
            for key, value in (metadata or {}).items():
                constant = getattr(cairo, "PDF_METADATA_" + {
                    "CreationDate": "CREATE_DATE", "ModDate": "MOD_DATE",
                }.get(key, key.upper()), None)
                if constant is not None:
                    surface.set_metadata(constant, "" if value is None else (
                        value.isoformat() if hasattr(value, "isoformat") else str(value)))
            context = cairo.Context(surface)
            context.scale(72 / dpi, 72 / dpi)
            renderer = RendererCairo(dpi)
            renderer.set_context(context)
            figure.draw(renderer)
            context.show_page()
        finally:
            surface.finish()


def savefig(figure, filename, *, compact_vector=False, **kwargs) -> None:
    """Export existing geometry with Heros lettering and Computer Modern math."""
    register_fonts()
    apply_fonts(figure)
    if compact_vector and Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Compact vector export requires a PDF output")
    if Path(filename).suffix.lower() == ".pdf":
        kwargs["backend"] = "pdf" if compact_vector else "module://paper_figure_style"
        kwargs.setdefault("dpi", 300)
    sizes = json.loads(Path(__file__).with_name("paper_figure_sizes.json").read_text())
    target = sizes.get(Path(filename).stem)
    original_bbox = figure.get_tightbbox
    if target and isinstance(kwargs.get("bbox_inches"), str) and kwargs["bbox_inches"] == "tight":
        pad = kwargs.get("pad_inches", mpl.rcParams["savefig.pad_inches"])

        def fixed_tightbbox(renderer, *args, **options):
            box = original_bbox(renderer, *args, **options)
            width, height = [points / 72 - 2 * pad for points in target]
            # Center the new text extents within the original physical canvas.
            # Axis limits, data marks and figure proportions stay unchanged.
            if box.width > width + .03 or box.height > height + .03:
                raise ValueError(f"{filename}: text exceeds the preserved export canvas")
            return Bbox.from_bounds(box.x0 - (width - box.width) / 2,
                                    box.y0 - (height - box.height) / 2, width, height)

        figure.get_tightbbox = fixed_tightbbox
    try:
        if compact_vector:
            # Reuse marker paths through the native PDF backend. Embed the CFF
            # lettering as Type 3 vector outlines, avoiding its Type 42 mislabel.
            with mpl.rc_context({"pdf.fonttype": 3}):
                figure.savefig(filename, **kwargs)
        else:
            figure.savefig(filename, **kwargs)
    finally:
        figure.get_tightbbox = original_bbox
