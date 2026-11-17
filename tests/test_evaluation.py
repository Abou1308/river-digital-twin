"""Class-level evaluation metrics."""

import numpy as np
import pytest

from rdt.monitoring import brier_score, class_scores


def test_brier_score_extremes():
    obs = np.array([0, 2])
    assert brier_score(np.eye(4)[obs], obs) == 0.0                       # certain and right
    assert brier_score(np.eye(4)[[1, 3]], obs) == 2.0                    # certain and wrong
    assert brier_score(np.full((2, 4), 0.25), obs) == pytest.approx(0.75)  # uninformed: 0.75^2 + 3 * 0.25^2


def test_class_scores():
    p = np.array([[0.1, 0.7, 0.2, 0.0],     # predicted slightly, observed slightly
                  [0.0, 0.6, 0.4, 0.0],     # predicted slightly, observed moderately
                  [0.0, 0.2, 0.8, 0.0]])    # predicted moderately, observed moderately
    s = class_scores(p, np.array([1, 2, 2]))
    assert s["n"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["confusion"].tolist() == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    assert s["observed_shares"].tolist() == pytest.approx([0, 1 / 3, 2 / 3, 0])
    # (0.01 + 0.09 + 0.04) + (0.36 + 0.36) + (0.04 + 0.04) = 0.94, over 3 samples
    assert s["brier"] == pytest.approx(0.94 / 3)
