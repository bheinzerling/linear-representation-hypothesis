from pathlib import Path
import hashlib
from PIL import Image
import claims_geometry as base
import claims_badges as old
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'data/figures/claims/prism.png'
COLORS = ['#917154', '#AD9573', '#695447', '#6A7880', '#949DA2', '#47545C']
CLAIM_ROWS = [151 + 47 * i for i in range(7)]
LAYOUTS = [dict(id='03-tighter-stack',height=490,art_y=60,rows=[94,142,190,292,346,400],representation_y=232,widths=[84,76,106,106,92,100])]
def point(layout, xy):
    return 350+xy[0]*620/1536, layout['art_y']+xy[1]*620/1536

def inset(c, index, y, width):
    """Scale diagram geometry uniformly; keep scientific text legible."""
    original_width, original_height = [(109, 51), (120, 62), (120, 58),
                                       (120, 58), (116, 56), (109, 55)][index]
    height = original_height*width/original_width
    x = 218+(120-width)/2
    if index == 0:
        base.direction(c, x, y-height/2, width, height, COLORS[index])
    elif index == 1:
        base.plane(c, x, y-height/2, width, height, COLORS[index])
    elif index in (2, 3):
        base.decode(c, x, y-height/2, width, height, COLORS[index])
    elif index == 4:
        base.relations(c, x, y-height/2, width, height, COLORS[index])
    else:
        # Keep the computation's full occupied width within the same inset column.
        base.computation(c, 218+(109-width)/2, y-height/2, width, height, COLORS[index])
    return [width, height]

def diagram(v, *, slug=None, save=True, badge_family='DejaVu Sans'):
    c = base.Canvas(slug or v['id']+'-figure', v['height'])
    image = Image.open(ART)
    assert image.size == (1536, 1024)
    extent = [350, 970, v['art_y']+620*2/3, v['art_y']]
    c.ax.imshow(image, extent=extent, zorder=0, aspect='auto', interpolation='lanczos')
    c.text(28, 43, 'Linear?', 31, weight='bold')
    c.text(28, v['representation_y'], 'Representation?', 31, weight='bold')
    dimensions = []
    for i, y in enumerate(v['rows']):
        c.text(28, y, old.LABELS[i], 22, va='center')
        dimensions.append(inset(c, i, y, v['widths'][i]))
        c.line([(354, y), point(v, old.VARIANTS[1]['inputs'][i])], COLORS[i], .65, alpha=.7)
    wx, wy = point(v, old.VARIANTS[1]['white'])
    c.text(wx, wy-55, 'the LRH', 25, weight='bold', ha='center')
    c.line([(wx, wy-22), (wx, wy-9)], base.MUTED, .6)
    c.text(947, 43, 'Hypothesis?', 31, weight='bold')
    c.text(947, 87, 'Spectrum of LRH claims', 19, base.MUTED)
    leaders = []
    for i, y in enumerate(CLAIM_ROWS):
        points = [point(v, old.VARIANTS[1]['outputs'][i]), (955, y), (963, y)]
        c.line(points, base.COLORS[i], .75)
        c.ax.lines[-1].set_gid('spectrum-leader-'+base.SHORT[i])
        old.ink_badge(c, 978, y, i, family=badge_family)
        c.text(1000, y, base.NAMES[i], 21, weight='bold', va='center')
        leaders.append(points)
    if save:
        c.save()
    audit = dict(variant=v['id'], linear_heading_y=43, hypothesis_heading_y=43,
                 artwork_extent=extent, artwork_dimensions=list(image.size),
                 artwork_sha256=hashlib.sha256(ART.read_bytes()).hexdigest(),
                 rows=v['rows'], symbol_dimensions=dimensions, spectrum_leaders=leaders,
                 optical_scene_translation_y=v['art_y']-150)
    return c, audit
