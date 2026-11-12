"""The TextRL utilities: option strings with nested component configs, and the argmax that breaks ties at random."""

from __future__ import annotations

import numpy as np
import pytest

from plyagents.textrl.utils.options import read_options, split_options
from plyagents.textrl.utils.selection import argmax


def test_options_split_at_top_level_commas_only():
    assert split_options("a=x,b=y[c=1,d=z[e=2]],f=3", "test") == {"a": "x", "b": "y[c=1,d=z[e=2]]", "f": "3"}
    assert split_options("", "test") == {}
    with pytest.raises(ValueError, match="Unbalanced"):
        split_options("a=y[c=1", "test")
    with pytest.raises(ValueError, match="Unbalanced"):
        split_options("a=y]", "test")
    with pytest.raises(ValueError, match="repeated"):
        split_options("a=1,a=2", "test")


def test_read_options_refuses_unknown_names():
    assert read_options("a=1", {"a", "b"}, "test") == {"a": "1"}
    with pytest.raises(ValueError, match="Unknown options"):
        read_options("c=1", {"a", "b"}, "test")


def test_argmax_takes_the_best_and_breaks_ties_at_random():
    rng = np.random.default_rng(0)
    assert {argmax([0.0, 1.0, 0.5], rng) for _ in range(20)} == {1}
    assert {argmax([1.0, 0.0, 1.0], rng) for _ in range(50)} == {0, 2}
