
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import literature_panels as candidates
import deletion_style as figure_style
import plot_causal_use_numbers as numerical
import plot_causal_use_refusal as refusal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/causal_use"
OUTPUT = ROOT / "figures/fig_causal_use"
FIGURE_SIZE = (16 / 2.54, 2.35)

