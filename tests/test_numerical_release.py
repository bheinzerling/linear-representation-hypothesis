from pathlib import Path

import pandas as pd

from lrh.analysis.figure7_matched import calculate, verify_manuscript

ROOT = Path(__file__).resolve().parents[1]


def test_released_figure7_edits_match_the_manuscript():
    result = calculate(ROOT / "data/figures/numerical_sufficiency_atlas/individual_prompt_fits")
    verify_manuscript((ROOT / "paper/paper.tex").read_text(), result)
    validation = result["probe_validation"].set_index("method")
    assert validation.loc[["linear", "circular_reg"], "probe_target_count"].eq(2000).all()
    assert validation.loc[["linear", "circular_reg"], "correct_sum_count"].eq(0).all()


def test_every_probe_has_complete_predictions_and_evaluation_intervals():
    rows = pd.read_csv(ROOT / "data/numerical/reproduced_test_predictions.csv")
    assert rows.groupby("architecture").size().eq(100).all() and len(rows) == 400
    intervals = pd.read_csv(ROOT / "data/numerical/probe_uncertainty.csv")
    assert intervals.low.notna().all() and intervals.high.notna().all()
