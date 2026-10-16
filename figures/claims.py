from pathlib import Path
import hashlib, subprocess
from matplotlib import font_manager
import claims_diagram as compact
FONT = 'TeX Gyre Heros'
HEADINGS = {'Linear?', 'Representation?', 'Hypothesis?'}
CLAIM_NAMES = ['Decodability','Efficiency','Robustness','Invariance','Structural\ncorrespondence','Necessity','Sufficiency']
CLAIM_KEYS = ['D','E','R','I','SC','N','S']
CLAIM_ROWS = [151 + 47 * i for i in range(7)]
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def register_fonts():
    """Find the installed font explicitly; fail instead of silently substituting."""
    records = []
    for style in ('Regular', 'Bold'):
        path = Path(subprocess.check_output(
            ['fc-match', '-f', '%{file}', f'{FONT}:style={style}'], text=True))
        font_manager.fontManager.addfont(path)
        props = font_manager.FontProperties(fname=path)
        if props.get_name() != FONT:
            raise RuntimeError(f'Install {FONT}; fontconfig found {props.get_name()}')
        records.append(dict(family=FONT, style=style, file=path.name, sha256=sha256(path)))
    return records
