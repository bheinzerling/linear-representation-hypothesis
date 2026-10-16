from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import claims_geometry as base
BADGE_AUDIT = []
LABELS = ['direction','linear subspace','affine decoder','Accessibility','Structure','Computation']
VARIANTS = [{}, dict(inputs=[(150,141),(150,267),(150,358),(150,596),(150,699),(150,798)],
outputs=[(1455,643),(1455,695),(1455,745),(1455,792),(1455,840),(1455,887),(1455,929)],white=(631,457))]
def ink_badge(c,x,y,index,*,family='DejaVu Sans'):
    color = base.COLORS[index]
    c.circle(x,y,12.5,color,color,.4)
    circle = c.ax.patches[-1]
    circle.set_gid('claim-badge-'+base.SHORT[index])
    glyph = TextPath((0,0),base.SHORT[index],size=20,prop=FontProperties(family=family,weight='bold'))
    bounds = glyph.get_extents()
    transform = Affine2D().scale(1,-1).translate(x-(bounds.x0+bounds.x1)/2,y+(bounds.y0+bounds.y1)/2)
    patch = PathPatch(glyph,transform=transform+c.ax.transData,facecolor='white',edgecolor='none',zorder=4)
    patch.set_gid('claim-initial-'+base.SHORT[index])
    c.ax.add_patch(patch)
    # Compare visible outline bounds in display space, independent of font baseline.
    cb = circle.get_window_extent()
    gb = patch.get_window_extent()
    error = max(abs((gb.x0+gb.x1-cb.x0-cb.x1)/2),abs((gb.y0+gb.y1-cb.y0-cb.y1)/2))
    if error > .01:
        raise ValueError(f'Badge {base.SHORT[index]} miscentered by {error} pixels')
    BADGE_AUDIT.append(dict(variant=c.slug,letter=base.SHORT[index],center_error_px=error))
