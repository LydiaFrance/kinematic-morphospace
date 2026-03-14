"""Tests for kinematic_morphospace.preprocessing.duplicate_resolution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinematic_morphospace.preprocessing.duplicate_resolution import (
    _replace_base_label,
    _strip_side_prefix,
    detect_duplicates,
    resolve_duplicates,
    split_labelled_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a marker DataFrame from a list of row dicts."""
    cols = ["frameID", "label", "xyz_1", "xyz_2", "xyz_3"]
    records = []
    for r in rows:
        record = {c: r.get(c, 0.0) for c in cols}
        record["frameID"] = r.get("frameID", "f1")
        record["label"] = r.get("label", "")
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TestDetectDuplicates
# ---------------------------------------------------------------------------

class TestDetectDuplicates:

    def test_all_unique(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f1", "label": "left_primary"},
        ])
        unique, dup_pairs, excess = detect_duplicates(df)
        assert len(unique) == 2
        assert len(dup_pairs) == 0
        assert len(excess) == 0

    def test_duplicate_pair(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f1", "label": "left_primary"},
        ])
        unique, dup_pairs, excess = detect_duplicates(df)
        assert len(unique) == 1
        assert unique.iloc[0]["label"] == "left_primary"
        assert len(dup_pairs) == 2
        assert len(excess) == 0

    def test_excess_three(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f1", "label": "left_wingtip"},
        ])
        unique, dup_pairs, excess = detect_duplicates(df)
        assert len(unique) == 0
        assert len(dup_pairs) == 0
        assert len(excess) == 3

    def test_different_frames_not_duplicates(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f2", "label": "left_wingtip"},
        ])
        unique, dup_pairs, excess = detect_duplicates(df)
        assert len(unique) == 2
        assert len(dup_pairs) == 0

    def test_unlabelled_always_unique(self):
        df = _make_df([
            {"frameID": "f1", "label": ""},
            {"frameID": "f1", "label": ""},
        ])
        unique, dup_pairs, excess = detect_duplicates(df)
        assert len(unique) == 2
        assert len(dup_pairs) == 0

    def test_empty_input(self):
        df = pd.DataFrame(columns=["frameID", "label", "xyz_1", "xyz_2", "xyz_3"])
        unique, dup_pairs, excess = detect_duplicates(df)
        assert len(unique) == 0
        assert len(dup_pairs) == 0
        assert len(excess) == 0

    def test_no_mutation(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip"},
            {"frameID": "f1", "label": "left_wingtip"},
        ])
        original = df.copy()
        detect_duplicates(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# TestResolveDuplicates
# ---------------------------------------------------------------------------

class TestResolveDuplicates:

    def test_wingtip_closer_becomes_primary(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip",
             "xyz_1": 0.1, "xyz_2": 0.1, "xyz_3": 0.1},  # closer
            {"frameID": "f1", "label": "left_wingtip",
             "xyz_1": 0.5, "xyz_2": 0.5, "xyz_3": 0.5},  # further
        ])
        result = resolve_duplicates(df)
        labels = set(result["label"])
        assert "left_primary" in labels
        assert "left_wingtip" in labels

    def test_wingtip_closer_becomes_secondary_if_y_below_threshold(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_wingtip",
             "xyz_1": 0.1, "xyz_2": -0.5, "xyz_3": 0.1},  # closer, y < -0.1
            {"frameID": "f1", "label": "left_wingtip",
             "xyz_1": 0.5, "xyz_2": 0.5, "xyz_3": 0.5},
        ])
        result = resolve_duplicates(df)
        labels = set(result["label"])
        assert "left_secondary" in labels
        assert "left_wingtip" in labels

    def test_wingtip_custom_threshold(self):
        df = _make_df([
            {"frameID": "f1", "label": "right_wingtip",
             "xyz_1": 0.1, "xyz_2": -0.05, "xyz_3": 0.1},  # closer, y = -0.05
            {"frameID": "f1", "label": "right_wingtip",
             "xyz_1": 0.5, "xyz_2": 0.5, "xyz_3": 0.5},
        ])
        # Default threshold (-0.1): y = -0.05 is above → primary
        result_default = resolve_duplicates(df)
        assert "right_primary" in set(result_default["label"])

        # Custom threshold (0.0): y = -0.05 is below → secondary
        result_custom = resolve_duplicates(df.copy(), wingtip_y_threshold=0.0)
        assert "right_secondary" in set(result_custom["label"])

    def test_primary_closer_by_y_becomes_secondary(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_primary",
             "xyz_1": 0.0, "xyz_2": 0.1, "xyz_3": 0.0},   # |y|=0.1, closer
            {"frameID": "f1", "label": "left_primary",
             "xyz_1": 0.0, "xyz_2": 0.5, "xyz_3": 0.0},
        ])
        result = resolve_duplicates(df)
        labels = set(result["label"])
        assert "left_secondary" in labels
        assert "left_primary" in labels

    def test_tailtip_further_from_midline_becomes_secondary(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_tailtip",
             "xyz_1": -0.1, "xyz_2": 0.0, "xyz_3": 0.0},  # |x|=0.1
            {"frameID": "f1", "label": "left_tailtip",
             "xyz_1": -0.5, "xyz_2": 0.0, "xyz_3": 0.0},  # |x|=0.5, further
        ])
        result = resolve_duplicates(df)
        labels = set(result["label"])
        assert "left_secondary" in labels
        assert "left_tailtip" in labels
        # The further one (idx 1) should be secondary
        assert result.loc[result.index[1], "label"] == "left_secondary"

    def test_secondary_further_from_midline_becomes_wingtip(self):
        df = _make_df([
            {"frameID": "f1", "label": "right_secondary",
             "xyz_1": 0.1, "xyz_2": 0.0, "xyz_3": 0.0},
            {"frameID": "f1", "label": "right_secondary",
             "xyz_1": 0.8, "xyz_2": 0.0, "xyz_3": 0.0},  # further
        ])
        result = resolve_duplicates(df)
        labels = set(result["label"])
        assert "right_wingtip" in labels
        assert "right_secondary" in labels

    def test_side_prefix_preserved(self):
        df = _make_df([
            {"frameID": "f1", "label": "right_wingtip",
             "xyz_1": 0.1, "xyz_2": 0.1, "xyz_3": 0.1},
            {"frameID": "f1", "label": "right_wingtip",
             "xyz_1": 0.5, "xyz_2": 0.5, "xyz_3": 0.5},
        ])
        result = resolve_duplicates(df)
        # All labels should still have right_ prefix
        assert all(l.startswith("right_") for l in result["label"])

    def test_empty_input(self):
        df = pd.DataFrame(columns=["frameID", "label", "xyz_1", "xyz_2", "xyz_3"])
        result = resolve_duplicates(df)
        assert len(result) == 0

    def test_unknown_base_label_unchanged(self):
        df = _make_df([
            {"frameID": "f1", "label": "left_unknown",
             "xyz_1": 0.1, "xyz_2": 0.1, "xyz_3": 0.1},
            {"frameID": "f1", "label": "left_unknown",
             "xyz_1": 0.5, "xyz_2": 0.5, "xyz_3": 0.5},
        ])
        result = resolve_duplicates(df)
        # Unknown type: both labels should stay unchanged
        assert all(l == "left_unknown" for l in result["label"])

    def test_no_prefix_wingtip(self):
        """Wingtip without left_/right_ prefix should still resolve."""
        df = _make_df([
            {"frameID": "f1", "label": "wingtip",
             "xyz_1": 0.1, "xyz_2": 0.1, "xyz_3": 0.1},
            {"frameID": "f1", "label": "wingtip",
             "xyz_1": 0.5, "xyz_2": 0.5, "xyz_3": 0.5},
        ])
        result = resolve_duplicates(df)
        labels = set(result["label"])
        assert "primary" in labels
        assert "wingtip" in labels


# ---------------------------------------------------------------------------
# TestSplitLabelledTable
# ---------------------------------------------------------------------------

class TestSplitLabelledTable:

    def test_basic_split(self):
        df = _make_df([
            {"label": "left_wingtip"},
            {"label": "left_primary"},
            {"label": "right_secondary"},
            {"label": "left_tailtip"},
            {"label": "headpack"},
            {"label": "backpack"},
            {"label": "tailpack"},
            {"label": ""},
        ])
        result = split_labelled_table(df)
        assert len(result["feather"]) == 4
        assert len(result["body"]) == 3
        assert len(result["unlabelled"]) == 1

    def test_no_overlap(self):
        df = _make_df([
            {"label": "left_wingtip"},
            {"label": "headpack"},
            {"label": ""},
            {"label": "right_tailtip"},
        ])
        result = split_labelled_table(df)
        total = len(result["feather"]) + len(result["body"]) + len(result["unlabelled"])
        assert total == len(df)

    def test_empty_input(self):
        df = pd.DataFrame(columns=["frameID", "label", "xyz_1", "xyz_2", "xyz_3"])
        result = split_labelled_table(df)
        assert len(result["feather"]) == 0
        assert len(result["body"]) == 0
        assert len(result["unlabelled"]) == 0

    def test_all_unlabelled(self):
        df = _make_df([
            {"label": ""},
            {"label": ""},
        ])
        result = split_labelled_table(df)
        assert len(result["unlabelled"]) == 2
        assert len(result["feather"]) == 0
        assert len(result["body"]) == 0

    def test_feather_labels_with_prefix(self):
        df = _make_df([
            {"label": "right_wingtip"},
            {"label": "left_primary"},
            {"label": "right_secondary"},
            {"label": "left_tailtip"},
        ])
        result = split_labelled_table(df)
        assert len(result["feather"]) == 4

    def test_unknown_labels_go_to_unlabelled(self):
        df = _make_df([
            {"label": "left_unknown"},
            {"label": "something_else"},
        ])
        result = split_labelled_table(df)
        assert len(result["unlabelled"]) == 2


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestStripSidePrefix:

    def test_left(self):
        assert _strip_side_prefix("left_wingtip") == "wingtip"

    def test_right(self):
        assert _strip_side_prefix("right_primary") == "primary"

    def test_no_prefix(self):
        assert _strip_side_prefix("backpack") == "backpack"

    def test_empty(self):
        assert _strip_side_prefix("") == ""


class TestReplaceBaseLabel:

    def test_left_prefix(self):
        assert _replace_base_label("left_wingtip", "primary") == "left_primary"

    def test_right_prefix(self):
        assert _replace_base_label("right_tailtip", "secondary") == "right_secondary"

    def test_no_prefix(self):
        assert _replace_base_label("wingtip", "primary") == "primary"

    def test_preserves_prefix_only(self):
        assert _replace_base_label("left_a", "b") == "left_b"
