
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/panel_sources"
WORK = ROOT / "out/panel-source-marking"
SVG = "http://www.w3.org/2000/svg"
NS = "{" + SVG + "}"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
SPEC = json.loads((DATA / "spec.json").read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(path):
    return ET.parse(path, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()


def capture():
    manifest_path = DATA / "input_manifest.json"
    if manifest_path.exists():
        raise SystemExit("Inputs are already recorded; review changes before recapturing.")
    manifest = {}
    for figure in SPEC["figures"]:
        original = ROOT / "figures" / (figure["stem"] + ".svg")
        source = original
        if figure["stem"] == "fig_decodability_efficiency":
            source = DATA / "decodability-input.svg"
            source.write_bytes(original.read_bytes())
        manifest[figure["stem"]] = {
            "path": source.relative_to(ROOT).as_posix(),
            "original": original.relative_to(ROOT).as_posix(),
            "sha256": digest(source),
            "panel_titles": panel_titles(parse(source), figure),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def font_file(style, condensed=False):
    family = "TeX Gyre Heros Cn" if condensed else "TeX Gyre Heros"
    path = Path(subprocess.check_output(
        ["fc-match", "-f", "%{file}", family + ":style=" + style], text=True
    ).strip())
    assert path.name == ("texgyreheroscn-" if condensed else "texgyreheros-") + style.lower() + ".otf", path
    return path


REGULAR = font_file("Regular")
BOLD = font_file("Bold")
SOURCE = font_file("Regular", condensed=True)
FONTS = {400: TTFont(REGULAR), 700: TTFont(BOLD)}
SOURCE_FONT = TTFont(SOURCE)


def width(text, size, weight=400, font=None):
    font = font or FONTS[700 if weight >= 600 else 400]
    mapping = font.getBestCmap()
    return sum(font["hmtx"].metrics[mapping[ord(char)]][0] for char in text) * size / font["head"].unitsPerEm


def ink_box(text, size, weight, x, baseline, anchor="start", bold_letter=False, font_override=None):
    """Measure glyph outlines: SVG getBBox includes the font's full ascender box."""
    offset, bounds = 0, []
    for index, char in enumerate(text):
        selected = 700 if bold_letter and index < 3 else weight
        font = font_override or FONTS[700 if selected >= 600 else 400]
        glyphs, mapping = font.getGlyphSet(), font.getBestCmap()
        name = mapping[ord(char)]
        factor = size / font["head"].unitsPerEm
        pen = BoundsPen(glyphs)
        glyphs[name].draw(pen)
        if pen.bounds:
            left, bottom, right, top = pen.bounds
            bounds.append((offset + left * factor, -top * factor,
                           offset + right * factor, -bottom * factor))
        offset += font["hmtx"].metrics[name][0] * factor
    if anchor == "end":
        x -= offset
    left = min(box[0] for box in bounds)
    top = min(box[1] for box in bounds)
    right = max(box[2] for box in bounds)
    bottom = max(box[3] for box in bounds)
    return json.dumps({"x": x + left, "y": baseline + top,
                       "width": right - left, "height": bottom - top})


def wrap_title(text, first_width, later_width, size, weight):
    words = text.split()
    lines = []
    while words:
        available = first_width if not lines else later_width
        line = words.pop(0)
        assert width(line, size, weight) <= available, (line, available)
        while words and width(line + " " + words[0], size, weight) <= available:
            line += " " + words.pop(0)
        lines.append(line)
    return lines


def title_text(group):
    texts = ["".join(t.itertext()) for t in group.findall(NS + "text")]
    if not texts:
        texts = [e.text or "" for e in group if e.tag is ET.Comment]
    return " ".join(" ".join(texts).split())


def panel_titles(source, figure):
    """Record source labels independently of the current display labels."""
    main = source.find(NS + "g[@id='figure_1']")
    assert main is not None
    titles = {}
    for panel in figure["panels"]:
        groups = [main.find(NS + "g[@id='" + name + "']") for name in panel["groups"]]
        assert all(group is not None for group in groups), panel
        titles[panel["letter"]] = " ".join(title_text(group) for group in groups)
    return titles


def style_except_fill(node):
    result = {}
    for part in node.attrib.get("style", "").split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            if key.strip() != "fill":
                result[key.strip()] = value.strip()
    return result


def body_signature(root, ignored):
    rows = []
    def walk(node):
        if node.attrib.get("id") in ignored:
            return
        rows.append((str(node.tag), sorted((k, v) for k, v in node.attrib.items() if k != "style"),
                     style_except_fill(node), node.text))
        for child in node:
            walk(child)
    walk(root)
    return rows


def transparent(path):
    style = path.attrib["style"]
    assert "fill: #ffffff" in style
    path.set("style", style.replace("fill: #ffffff", "fill: none"))


def render(source, figure, source_titles=None):
    if source_titles is not None:
        assert set(source_titles) == {panel["letter"] for panel in figure["panels"]}
    root = copy.deepcopy(source)
    dimensions = root.attrib["viewBox"].split()
    native_width, native_height = map(float, dimensions[2:])
    scale = native_width / (SPEC["paper_width_cm"] / 2.54 * 72)
    source_size = SPEC["source_font_pt"] * scale
    main = root.find(NS + "g[@id='figure_1']")
    assert main is not None
    ignored = {name for panel in figure["panels"] for name in panel["groups"]}
    before = body_signature(main, ignored)
    kept_defs = ET.Element(NS + "defs")
    for panel in figure["panels"]:
        groups = [main.find(NS + "g[@id='" + name + "']") for name in panel["groups"]]
        assert all(group is not None for group in groups), panel
        actual = " ".join(title_text(group) for group in groups)
        expected = panel["title"] if source_titles is None else source_titles[panel["letter"]]
        assert actual == expected, (actual, expected)
        for group in groups:
            for definitions in group.iter(NS + "defs"):
                for definition in definitions:
                    kept_defs.append(copy.deepcopy(definition))
            main.remove(group)
    changed_backgrounds = []
    background = main.find(NS + "g[@id='patch_1']/" + NS + "path")
    transparent(background)
    changed_backgrounds.append("patch_1")
    for axes in main.iter(NS + "g"):
        if not axes.attrib.get("id", "").startswith("axes_") or not len(axes):
            continue
        first = axes[0]
        if first.attrib.get("id", "").startswith("patch_"):
            path = first.find(NS + "path")
            if path is not None and path.attrib.get("style", "").strip() == "fill: #ffffff":
                transparent(path)
                changed_backgrounds.append(first.attrib["id"])
    assert body_signature(main, ignored) == before, "Body geometry, text, or non-fill style changed"
    for name, (dx, dy) in figure.get("body_translation_pt", {}).items():
        axes = main.find(NS + "g[@id='" + name + "']")
        assert axes is not None
        position = list(main).index(axes)
        main.remove(axes)
        wrapper = ET.Element(NS + "g", {"transform": f"translate({dx} {dy})", "data-body-translation": name})
        wrapper.append(axes)
        main.insert(position, wrapper)
    root.remove(main)
    root.append(kept_defs)
    ET.SubElement(root, NS + "rect", {"width": str(native_width), "height": str(native_height), "fill": "#ffffff"})
    underlay = ET.SubElement(root, NS + "g", {"id": "panel-source-backgrounds"})
    for panel in figure["panels"]:
        colours = SPEC["palette"][panel["source"]]
        assert 0 <= panel["left"] < panel["right"] <= native_width
        assert 0 <= panel["top"] < panel["bottom"] <= native_height
        ET.SubElement(underlay, NS + "rect", {
            "x": str(panel["left"]), "y": str(panel["top"]),
            "width": str(panel["right"] - panel["left"]),
            "height": str(panel["bottom"] - panel["top"]), "fill": colours["tint"],
        })
    root.append(main)
    headers = ET.SubElement(root, NS + "g", {"id": "panel-source-headers"})
    records = []
    for panel in figure["panels"]:
        colours = SPEC["palette"][panel["source"]]
        right = panel["right"] - 3 * scale
        label_left = right - width(colours["label"], source_size, font=SOURCE_FONT)
        gap = 2 * scale
        title_size = figure["title_size"]
        first_width = label_left - panel["title_x"] - gap
        later_width = right - panel["title_x"]
        lines = panel.get("title_lines") or wrap_title(
            panel["title"], first_width, later_width, title_size, figure["title_weight"])
        assert " ".join(lines) == panel["title"]
        for index, line in enumerate(lines):
            assert width(line, title_size, figure["title_weight"]) <= (first_width if index == 0 else later_width)
        baseline = panel["top"] + 2.5 * scale + .8 * title_size
        group = ET.SubElement(headers, NS + "g", {"data-panel-letter": panel["letter"], "data-source": panel["source"]})
        ET.SubElement(group, NS + "rect", {
            "x": str(panel["left"]), "y": str(panel["top"]),
            "width": str(panel["right"] - panel["left"]),
            "height": str(panel["bottom"] - panel["top"]), "fill": "none",
            "stroke": colours["ink"], "stroke-width": str(.65 * scale),
        })
        for index, line in enumerate(lines):
            line_baseline = baseline + index * title_size * 1.08
            text = ET.SubElement(group, NS + "text", {
                "x": str(panel["title_x"]), "y": str(line_baseline),
                "style": f"font-family:'TeX Gyre Heros';font-size:{title_size}px;font-weight:{figure['title_weight']};font-kerning:none;fill:{colours['ink']}",
                "data-role": "panel-title",
                "data-ink-box": ink_box(line, title_size, figure["title_weight"],
                                        panel["title_x"], line_baseline,
                                        bold_letter=index == 0 and figure["title_weight"] == 400),
            })
            if index == 0 and figure["title_weight"] == 400:
                letter, rest = line.split(" ", 1) if " " in line else (line, "")
                ET.SubElement(text, NS + "tspan", {"style": "font-weight:700"}).text = letter
                if rest:
                    ET.SubElement(text, NS + "tspan").text = " " + rest
            else:
                text.text = line
        ET.SubElement(group, NS + "text", {
            "x": str(right), "y": str(baseline - .15 * scale), "text-anchor": "end",
            "style": f"font-family:'TeX Gyre Heros Cn';font-size:{source_size}px;font-weight:400;font-kerning:none;fill:{colours['ink']}",
            "data-role": "source-label",
            "data-ink-box": ink_box(colours["label"], source_size, 400, right,
                                    baseline - .15 * scale, anchor="end", font_override=SOURCE_FONT),
        }).text = colours["label"]
        records.append({"letter": panel["letter"], "source": panel["source"], "title_lines": lines})
    assert root.attrib["viewBox"] == source.attrib["viewBox"]
    return ET.tostring(root, encoding="unicode"), {
        "body_geometry_text_and_non_fill_styles": "identical",
        "body_translation_pt": figure.get("body_translation_pt", {}),
        "canvas_size": "identical", "transparent_backgrounds": changed_backgrounds,
        "panels": records, "width_pt": native_width, "height_pt": native_height,
    }


def browser_font(path):
    """Use TrueType outlines so Chromium embeds text without Type 3 fonts."""
    if path.suffix != ".otf":
        return path
    original = TTFont(path)
    builder = FontBuilder(original["head"].unitsPerEm, isTTF=True)
    builder.setupGlyphOrder(original.getGlyphOrder())
    builder.setupCharacterMap(original.getBestCmap())
    glyph_set, glyphs = original.getGlyphSet(), {}
    for name in original.getGlyphOrder():
        pen = TTGlyphPen(None)
        glyph_set[name].draw(Cu2QuPen(pen, max_err=.5, reverse_direction=True))
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(original["hmtx"].metrics)
    builder.setupHorizontalHeader(ascent=original["hhea"].ascent,
                                  descent=original["hhea"].descent,
                                  lineGap=original["hhea"].lineGap)
    builder.font["name"] = copy.deepcopy(original["name"])
    builder.font["head"].macStyle = original["head"].macStyle
    os2 = original["OS/2"]
    builder.setupOS2(**{name: getattr(os2, name) for name in
                      ("sTypoAscender", "sTypoDescender", "sTypoLineGap", "usWinAscent",
                       "usWinDescent", "usWeightClass", "usWidthClass", "fsSelection", "fsType")})
    builder.setupPost()
    builder.setupMaxp()
    builder.font.recalcTimestamp = False
    for name in ("created", "modified"):
        setattr(builder.font["head"], name, getattr(original["head"], name))
    output = WORK / (path.stem + ".ttf")
    builder.save(output)
    return output


def fonts_css():
    sources = [("TeX Gyre Heros", "100 500", REGULAR), ("TeX Gyre Heros", "600 900", BOLD),
               ("TeX Gyre Heros Cn", "400", SOURCE)]
    for name in ("cmr10", "cmmi10", "cmsy10"):
        sources.append((name, "400", Path(matplotlib.get_data_path()) / "fonts/ttf" / (name + ".ttf")))
    sources.append(("STIXGeneral", "400", Path(matplotlib.get_data_path()) / "fonts/ttf/STIXGeneral.ttf"))
    css, records = [], []
    for family, weight, path in sources:
        rendered = browser_font(path)
        css.append("@font-face{font-family:'" + family + "';font-weight:" + weight
                   + ";src:url(data:font/ttf;base64," + base64.b64encode(rendered.read_bytes()).decode()
                   + ") format('truetype');font-display:block;}")
        records.append({"family": family, "weight": weight, "path": str(path), "sha256": digest(path),
                        "browser_font_sha256": digest(rendered)})
    return "\n".join(css), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-inputs", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.capture_inputs:
        capture()
    manifest = json.loads((DATA / "input_manifest.json").read_text())
    output_records = []
    for figure in SPEC["figures"]:
        item = manifest[figure["stem"]]
        source = ROOT / item["path"]
        assert digest(source) == item["sha256"], "Base figure changed: " + str(source)
        svg, check = render(parse(source), figure, item.get("panel_titles"))
        output = ROOT / "figures" / (figure["stem"] + "_source_marked.svg")
        output.write_text(svg + "\n")
        output_records.append({"stem": figure["stem"], "svg": str(output.relative_to(ROOT)), "check": check})
    css, fonts = fonts_css()
    (WORK / "fonts.css").write_text(css)
    (WORK / "render-job.json").write_text(json.dumps({"figures": output_records, "fonts_css": str(WORK / "fonts.css")}, indent=2))
    subprocess.run(["node", str(ROOT / "figures/export_source_marked.cjs")], cwd=ROOT, check=True)
    browser_checks = json.loads((WORK / "browser-checks.json").read_text())
    provenance = {
        "command": "uv run python figures/gen_fig_source_marked.py",
        "generator_sha256": digest(__file__), "exporter_sha256": digest(ROOT / "figures/export_source_marked.cjs"),
        "spec_sha256": digest(DATA / "spec.json"), "inputs": manifest, "fonts": fonts,
        "figures": output_records, "browser_checks": browser_checks,
        "exports": {str(path.relative_to(ROOT)): digest(path)
                    for figure in SPEC["figures"] for ext in ("svg", "pdf")
                    for path in [ROOT / "figures" / (figure["stem"] + "_source_marked." + ext)]},
    }
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figures": len(output_records), "checks": [record["check"] for record in output_records]}, indent=2))


if __name__ == "__main__":
    main()
