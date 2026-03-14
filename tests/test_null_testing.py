"""
test_null_testing.py

Tests for the null_testing module: flatten_frames, ensure_rng,
validate_frame_alignment, prepare_sequence_groups, sequence_lookup,
grouped_bootstrap_indices, grouped_permutation_labels,
summarise_distribution, summarise_cumulative_variance,
pairwise_distance_features, principal_cosines,
random_relabel_frames, relabel_with_predictor.
"""

import numpy as np
import pandas as pd
import pytest

from kinematic_morphospace.null_testing import (
    flatten_frames,
    ensure_rng,
    validate_frame_alignment,
    prepare_sequence_groups,
    sequence_lookup,
    grouped_bootstrap_indices,
    grouped_permutation_labels,
    summarise_distribution,
    summarise_cumulative_variance,
    pairwise_distance_features,
    principal_cosines,
    random_relabel_frames,
    relabel_with_predictor,
)


# -- Fixtures --

@pytest.fixture
def frame_info_df():
    """Minimal frame_info DataFrame with required columns."""
    return pd.DataFrame({
        "seqID": ["s1", "s1", "s2", "s2", "s2"],
        "BirdID": ["B1", "B1", "B2", "B2", "B2"],
        "Obstacle": [0, 0, 1, 1, 1],
        "Left": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def frames_3d():
    """Small synthetic (5 frames, 4 markers, 3 dims) array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 4, 3))


# =============================================================================
# flatten_frames
# =============================================================================

class TestFlattenFrames:
    def test_3d_to_2d_reshape(self, frames_3d):
        result = flatten_frames(frames_3d)
        assert result.ndim == 2
        assert result.shape == (5, 12)

    def test_2d_passthrough_copy(self):
        arr = np.ones((5, 12))
        result = flatten_frames(arr)
        assert result.shape == arr.shape
        # Must be a copy, not the same object
        assert result is not arr
        np.testing.assert_array_equal(result, arr)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="Unexpected frame array shape"):
            flatten_frames(np.ones(10))


# =============================================================================
# ensure_rng
# =============================================================================

class TestEnsureRng:
    def test_integer_seed_deterministic(self):
        rng1 = ensure_rng(42)
        rng2 = ensure_rng(42)
        assert rng1.random() == rng2.random()

    def test_none_returns_generator(self):
        rng = ensure_rng(None)
        assert isinstance(rng, np.random.Generator)

    def test_passthrough_existing_generator(self):
        original = np.random.default_rng(99)
        returned = ensure_rng(original)
        assert returned is original


# =============================================================================
# validate_frame_alignment
# =============================================================================

class TestValidateFrameAlignment:
    def test_valid_passes(self, frames_3d, frame_info_df):
        # Should not raise
        validate_frame_alignment(frames_3d, frame_info_df)

    def test_length_mismatch_raises(self, frame_info_df):
        bad_frames = np.zeros((10, 4, 3))  # 10 != 5
        with pytest.raises(ValueError, match="Frame count mismatch"):
            validate_frame_alignment(bad_frames, frame_info_df)

    def test_missing_columns_raises(self, frames_3d):
        df = pd.DataFrame({"seqID": range(5), "BirdID": range(5)})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_frame_alignment(frames_3d, df)


# =============================================================================
# prepare_sequence_groups
# =============================================================================

class TestPrepareSequenceGroups:
    def test_expected_dict_keys(self, frame_info_df):
        groups = prepare_sequence_groups(frame_info_df)
        expected_keys = {
            "seq_ids", "unique_seq_ids", "seq_index",
            "side_flag", "bird_ids", "obstacles",
        }
        assert set(groups.keys()) == expected_keys

    def test_seq_index_maps_back(self, frame_info_df):
        groups = prepare_sequence_groups(frame_info_df)
        reconstructed = groups["unique_seq_ids"][groups["seq_index"]]
        np.testing.assert_array_equal(reconstructed, groups["seq_ids"])


# =============================================================================
# sequence_lookup
# =============================================================================

class TestSequenceLookup:
    def test_correct_group_count(self):
        seq_index = np.array([0, 0, 1, 1, 1])
        lookup = sequence_lookup(seq_index)
        assert len(lookup) == 2

    def test_indices_cover_all_frames(self):
        seq_index = np.array([0, 0, 1, 1, 1])
        lookup = sequence_lookup(seq_index)
        all_indices = sorted(np.concatenate(lookup).tolist())
        assert all_indices == [0, 1, 2, 3, 4]


# =============================================================================
# grouped_bootstrap_indices
# =============================================================================

class TestGroupedBootstrapIndices:
    def test_returns_bool_mask(self):
        seq_index = np.array([0, 0, 1, 1, 2, 2])
        mask = grouped_bootstrap_indices(seq_index, seed=42)
        assert mask.dtype == bool
        assert mask.shape == (6,)

    def test_deterministic_with_seed(self):
        seq_index = np.array([0, 0, 1, 1, 2, 2])
        mask1 = grouped_bootstrap_indices(seq_index, seed=42)
        mask2 = grouped_bootstrap_indices(seq_index, seed=42)
        np.testing.assert_array_equal(mask1, mask2)

    def test_whole_sequences_resampled_together(self):
        seq_index = np.array([0, 0, 1, 1, 2, 2])
        mask = grouped_bootstrap_indices(seq_index, seed=42)
        # For each sequence, either all frames are selected or none
        for seq_id in np.unique(seq_index):
            seq_mask = mask[seq_index == seq_id]
            assert np.all(seq_mask) or not np.any(seq_mask)


# =============================================================================
# grouped_permutation_labels
# =============================================================================

class TestGroupedPermutationLabels:
    def test_deterministic_with_seed(self):
        labels = np.array([0, 0, 1, 1, 1])
        seq_index = np.array([0, 0, 1, 1, 1])
        perm1 = grouped_permutation_labels(labels, seq_index, seed=42)
        perm2 = grouped_permutation_labels(labels, seq_index, seed=42)
        np.testing.assert_array_equal(perm1, perm2)

    def test_uniform_labels_within_sequences(self):
        labels = np.array([0, 0, 1, 1, 1])
        seq_index = np.array([0, 0, 1, 1, 1])
        perm = grouped_permutation_labels(labels, seq_index, seed=42)
        # Within each sequence all frames should have same label
        for seq_id in np.unique(seq_index):
            vals = perm[seq_index == seq_id]
            assert np.all(vals == vals[0])


# =============================================================================
# summarise_distribution
# =============================================================================

class TestSummariseDistribution:
    def test_returns_series_with_expected_keys(self):
        values = np.arange(100, dtype=float)
        result = summarise_distribution(values)
        assert isinstance(result, pd.Series)
        assert "mean" in result.index
        assert "std" in result.index
        assert "p2.5" in result.index
        assert "p50" in result.index
        assert "p97.5" in result.index


# =============================================================================
# summarise_cumulative_variance
# =============================================================================

class TestSummariseCumulativeVariance:
    def test_returns_dataframe_with_k_index(self):
        cev = np.random.default_rng(42).random((50, 5))
        result = summarise_cumulative_variance(cev)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == [f"k={i}" for i in range(1, 6)]


# =============================================================================
# pairwise_distance_features
# =============================================================================

class TestPairwiseDistanceFeatures:
    def test_output_shape(self, frames_3d):
        result = pairwise_distance_features(frames_3d)
        # 4 markers → 4*3/2 = 6 pairs
        assert result.shape == (5, 6)

    def test_descending_sort(self, frames_3d):
        result = pairwise_distance_features(
            frames_3d, sort_per_frame=True, descending=True
        )
        for row in result:
            np.testing.assert_array_equal(row, np.sort(row)[::-1])

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="Expected frames with shape"):
            pairwise_distance_features(np.ones((5, 12)))


# =============================================================================
# principal_cosines
# =============================================================================

class TestPrincipalCosines:
    def test_identical_bases_give_ones(self):
        rng = np.random.default_rng(42)
        basis = np.linalg.qr(rng.standard_normal((10, 3)))[0]
        cosines = principal_cosines(basis, basis, modes=3)
        np.testing.assert_allclose(cosines, 1.0, atol=1e-12)

    def test_orthogonal_bases_give_zeros(self):
        # Two orthogonal 2-D subspaces of R^4
        A = np.eye(4)[:, :2]  # spans e1, e2
        B = np.eye(4)[:, 2:]  # spans e3, e4
        cosines = principal_cosines(A, B, modes=2)
        np.testing.assert_allclose(cosines, 0.0, atol=1e-12)

    def test_known_partial_alignment(self):
        """A 45° rotation in a 2-D subspace of R^6 should give
        principal cosines [cos(45°), 1] — one aligned and one rotated."""
        # Basis A: first 2 columns of I₆
        A = np.eye(6)[:, :2]

        # Rotate the first basis vector 45° toward the third axis,
        # keep the second basis vector unchanged.
        angle = np.radians(45)
        B = np.zeros((6, 2))
        B[0, 0] = np.cos(angle)  # e1 rotated toward e3
        B[2, 0] = np.sin(angle)
        B[1, 1] = 1.0            # e2 unchanged

        cosines = principal_cosines(A, B, modes=2)
        # SVD returns principal cosines in descending order
        expected = sorted([np.cos(angle), 1.0], reverse=True)
        np.testing.assert_allclose(cosines, expected, atol=1e-12)

    def test_return_angles_flag(self):
        rng = np.random.default_rng(42)
        basis = np.linalg.qr(rng.standard_normal((10, 3)))[0]
        result = principal_cosines(basis, basis, modes=3, return_angles=True)
        assert isinstance(result, tuple)
        cosines, angles = result
        assert cosines.shape == (3,)
        assert angles.shape == (3,)
        np.testing.assert_allclose(angles, 0.0, atol=1e-6)


# =============================================================================
# random_relabel_frames
# =============================================================================

class TestRandomRelabelFrames:
    def test_shape_preserved(self, frames_3d):
        rng = np.random.default_rng(42)
        result = random_relabel_frames(frames_3d, swap_fraction=0.5, rng=rng)
        assert result.shape == frames_3d.shape

    def test_swap_fraction_zero_no_change(self, frames_3d):
        rng = np.random.default_rng(42)
        result = random_relabel_frames(frames_3d, swap_fraction=0.0, rng=rng)
        np.testing.assert_array_equal(result, frames_3d)


# =============================================================================
# Structured vs shuffled CEV (R5)
# =============================================================================

class TestStructuredVsShuffledCEV:
    """S8.2: Structured (correlated) data should have much higher CEV₄
    than column-shuffled data where correlations are destroyed."""

    def test_structured_data_has_higher_cev_than_shuffled(self):
        """Synthetic data with a single latent factor should yield CEV₄ >>
        shuffled CEV₄, confirming that PCA detects genuine structure."""
        from kinematic_morphospace.null_testing import flatten_frames
        from kinematic_morphospace.pca_core import run_PCA

        rng = np.random.default_rng(42)
        n_frames, n_markers, n_dims = 200, 4, 3

        # Build correlated data: single latent factor + noise
        latent = rng.standard_normal((n_frames, 1))
        loadings = rng.standard_normal((1, n_markers * n_dims))
        structured_2d = latent @ loadings + 0.3 * rng.standard_normal(
            (n_frames, n_markers * n_dims)
        )
        structured_3d = structured_2d.reshape(n_frames, n_markers, n_dims)

        # CEV₄ on structured data
        _, _, pca_structured = run_PCA(structured_3d)
        cev4_structured = np.sum(pca_structured.explained_variance_ratio_[:4])

        # Column-shuffle to destroy correlations
        flat = flatten_frames(structured_3d)
        shuffled = flat.copy()
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])
        shuffled_3d = shuffled.reshape(n_frames, n_markers, n_dims)

        _, _, pca_shuffled = run_PCA(shuffled_3d)
        cev4_shuffled = np.sum(pca_shuffled.explained_variance_ratio_[:4])

        assert cev4_structured > cev4_shuffled + 0.1, (
            f"Structured CEV₄ ({cev4_structured:.3f}) should exceed "
            f"shuffled CEV₄ ({cev4_shuffled:.3f}) by a meaningful margin"
        )


# =============================================================================
# relabel_with_predictor
# =============================================================================

class TestRelabelWithPredictor:
    def test_first_frame_preserved(self, frames_3d):
        result = relabel_with_predictor(frames_3d)
        np.testing.assert_array_equal(result[0], frames_3d[0])
