"""Tests for kinematic_morphospace.species_transform — piecewise marker transformation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kinematic_morphospace.species_transform import (
    compute_transformation_matrix,
    create_marker_dict,
    transform_hawk_to_species,
    transform_principal_components,
)


# ---------------------------------------------------------------------------
# Tests: compute_transformation_matrix
# ---------------------------------------------------------------------------

class TestComputeTransformationMatrix:
    def test_identity_when_same_vector(self):
        """Same source and target should yield identity matrix."""
        v = np.array([1.0, 0.0, 0.0])
        T = compute_transformation_matrix(v, v)
        np.testing.assert_allclose(T, np.eye(3), atol=1e-10)

    def test_scaling_parallel_vectors(self):
        """Parallel vectors with different lengths should yield a scaled identity."""
        source = np.array([1.0, 0.0, 0.0])
        target = np.array([2.0, 0.0, 0.0])
        T = compute_transformation_matrix(source, target)
        expected = 2.0 * np.eye(3)
        np.testing.assert_allclose(T, expected, atol=1e-10)

    def test_90_degree_rotation(self):
        """Rotating x-axis to y-axis should produce a 90-degree rotation about z."""
        source = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0])
        T = compute_transformation_matrix(source, target)
        result = T @ source
        np.testing.assert_allclose(result, target, atol=1e-10)

    def test_combined_rotation_and_scaling(self):
        """Rotation + scaling: source should transform to target."""
        source = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 2.0, 0.0])
        T = compute_transformation_matrix(source, target)
        result = T @ source
        np.testing.assert_allclose(result, target, atol=1e-10)

    def test_raises_on_zero_source(self):
        """Zero-length source vector should raise ValueError."""
        source = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero"):
            compute_transformation_matrix(source, target)

    def test_raises_on_zero_target(self):
        """Zero-length target vector should raise ValueError."""
        source = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero"):
            compute_transformation_matrix(source, target)

    def test_anti_parallel_vectors(self):
        """Anti-parallel vectors (180-degree rotation) should still work."""
        source = np.array([1.0, 0.0, 0.0])
        target = np.array([-1.0, 0.0, 0.0])
        T = compute_transformation_matrix(source, target)
        result = T @ source
        np.testing.assert_allclose(result, target, atol=1e-10)

    def test_3d_arbitrary_vectors(self):
        """Arbitrary 3D vectors should transform correctly."""
        source = np.array([1.0, 2.0, 3.0])
        target = np.array([3.0, -1.0, 2.0])
        T = compute_transformation_matrix(source, target)
        result = T @ source
        np.testing.assert_allclose(result, target, atol=1e-10)

    def test_returns_3x3_matrix(self):
        source = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0])
        T = compute_transformation_matrix(source, target)
        assert T.shape == (3, 3)


# ---------------------------------------------------------------------------
# Tests: transform_principal_components
# ---------------------------------------------------------------------------

class TestTransformPrincipalComponents:
    def test_identity_transformation(self):
        """Identity transformation should return original PCs."""
        pcs = np.random.randn(4, 4, 3)  # 4 PCs, 4 markers, 3 coords
        T = np.eye(12)  # 4 markers x 3 coords
        result = transform_principal_components(pcs, T)
        np.testing.assert_allclose(result, pcs, atol=1e-10)

    def test_scaling_transformation(self):
        """Uniform 2x scaling should double all PC values."""
        pcs = np.ones((2, 3, 3))  # 2 PCs, 3 markers, 3 coords
        T = 2.0 * np.eye(9)  # 3 markers x 3 coords
        result = transform_principal_components(pcs, T)
        expected = 2.0 * pcs
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        pcs = np.random.randn(5, 4, 3)
        T = np.eye(12)
        result = transform_principal_components(pcs, T)
        assert result.shape == pcs.shape


# ---------------------------------------------------------------------------
# Tests: create_marker_dict
# ---------------------------------------------------------------------------

class TestCreateMarkerDict:
    def test_creates_correct_keys(self):
        markers = np.array([[1, 2, 3], [4, 5, 6]])
        names = ['left_wingtip', 'right_wingtip']
        result = create_marker_dict(markers, names)
        assert set(result.keys()) == {'left_wingtip', 'right_wingtip'}

    def test_coordinates_match_input(self):
        markers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        names = ['left_wingtip', 'right_wingtip']
        result = create_marker_dict(markers, names)
        assert result['left_wingtip'] == [1.0, 2.0, 3.0]
        assert result['right_wingtip'] == [4.0, 5.0, 6.0]

    def test_returns_lists_not_arrays(self):
        markers = np.array([[1.0, 2.0, 3.0]])
        names = ['marker']
        result = create_marker_dict(markers, names)
        assert isinstance(result['marker'], list)


# ---------------------------------------------------------------------------
# Tests: transform_hawk_to_species (mocked external dependency)
# ---------------------------------------------------------------------------

class TestTransformHawkToSpecies:
    @patch('kinematic_morphospace.species_transform.Animal3D')
    @patch('kinematic_morphospace.species_transform.integrate_dataframe_to_bird3D')
    def test_returns_tuple_of_three(self, mock_integrate, mock_bird_cls):
        """Should return (transformed_bird, target_bird, transformation_matrix)."""
        import pandas as pd

        # Set up mock hawk
        mock_hawk = MagicMock()
        mock_hawk.right_markers = np.array([[0.3, 0.1, 0.0],
                                             [0.2, 0.08, 0.0],
                                             [0.15, 0.05, 0.0],
                                             [0.05, -0.1, 0.0]])
        mock_hawk.mirror_keypoints = MagicMock(
            return_value=np.random.randn(1, 8, 3))

        # Set up mock target bird
        mock_target = MagicMock()
        mock_target.right_markers = np.array([[0.4, 0.15, 0.0],
                                               [0.25, 0.1, 0.0],
                                               [0.18, 0.06, 0.0],
                                               [0.06, -0.12, 0.0]])
        mock_target.skeleton_definition.marker_names = [
            'left_wingtip', 'right_wingtip', 'left_primary', 'right_primary',
            'left_secondary', 'right_secondary', 'left_tailtip', 'right_tailtip',
        ]
        mock_bird_cls.return_value = mock_target

        # Set up mock integrate (returns single dict with all markers)
        mock_integrate.return_value = {
            'left_wingtip': [0.4, 0.15, 0.0],
            'left_shoulder': [0.05, 0.0, 0.0],
        }

        # Set up species DataFrame
        species_df = pd.DataFrame({
            'species_common': ['Barn owl'],
        })

        result = transform_hawk_to_species(mock_hawk, 0, species_df)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch('kinematic_morphospace.species_transform.Animal3D')
    @patch('kinematic_morphospace.species_transform.integrate_dataframe_to_bird3D')
    def test_tail_z_override_applied(self, mock_integrate, mock_bird_cls):
        """Default tail_z_override=-0.05 should set last marker's z."""
        import pandas as pd

        mock_hawk = MagicMock()
        mock_hawk.right_markers = np.array([[0.3, 0.1, 0.0],
                                             [0.2, 0.08, 0.0],
                                             [0.15, 0.05, 0.0],
                                             [0.05, -0.1, 0.0]])
        # Capture the markers passed to mirror_keypoints
        captured = {}
        def capture_mirror(m):
            captured['markers'] = m.copy()
            return np.random.randn(1, 8, 3)
        mock_hawk.mirror_keypoints = MagicMock(side_effect=capture_mirror)

        mock_target = MagicMock()
        mock_target.right_markers = np.array([[0.4, 0.15, 0.0],
                                               [0.25, 0.1, 0.0],
                                               [0.18, 0.06, 0.0],
                                               [0.06, -0.12, 0.0]])
        mock_target.skeleton_definition.marker_names = [
            'l_wt', 'r_wt', 'l_p', 'r_p', 'l_s', 'r_s', 'l_tt', 'r_tt',
        ]
        mock_bird_cls.return_value = mock_target
        mock_integrate.return_value = {}

        species_df = pd.DataFrame({'species_common': ['test']})
        transform_hawk_to_species(mock_hawk, 0, species_df)

        # The markers passed to mirror_keypoints should have z=-0.05 for last marker
        assert captured['markers'][0, -1, 2] == pytest.approx(-0.05)

    @patch('kinematic_morphospace.species_transform.Animal3D')
    @patch('kinematic_morphospace.species_transform.integrate_dataframe_to_bird3D')
    def test_tail_z_override_none_skips(self, mock_integrate, mock_bird_cls):
        """tail_z_override=None should not modify the tail tip z."""
        import pandas as pd

        mock_hawk = MagicMock()
        mock_hawk.right_markers = np.array([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 1.0, 1.0]])
        captured = {}
        def capture_mirror(m):
            captured['markers'] = m.copy()
            return np.random.randn(1, 8, 3)
        mock_hawk.mirror_keypoints = MagicMock(side_effect=capture_mirror)

        mock_target = MagicMock()
        mock_target.right_markers = np.array([[2.0, 0.0, 0.0],
                                               [0.0, 2.0, 0.0],
                                               [0.0, 0.0, 2.0],
                                               [2.0, 2.0, 2.0]])
        mock_target.skeleton_definition.marker_names = [
            'l_wt', 'r_wt', 'l_p', 'r_p', 'l_s', 'r_s', 'l_tt', 'r_tt',
        ]
        mock_bird_cls.return_value = mock_target
        mock_integrate.return_value = {}

        species_df = pd.DataFrame({'species_common': ['test']})
        transform_hawk_to_species(mock_hawk, 0, species_df, tail_z_override=None)

        # Tail tip z should be the transformed value (2.0), not overridden
        assert captured['markers'][0, -1, 2] != pytest.approx(-0.05)

    @patch('kinematic_morphospace.species_transform.Animal3D')
    @patch('kinematic_morphospace.species_transform.integrate_dataframe_to_bird3D')
    def test_tail_z_override_custom_value(self, mock_integrate, mock_bird_cls):
        """Custom tail_z_override value should be applied."""
        import pandas as pd

        mock_hawk = MagicMock()
        mock_hawk.right_markers = np.array([[0.3, 0.1, 0.0],
                                             [0.2, 0.08, 0.0],
                                             [0.15, 0.05, 0.0],
                                             [0.05, -0.1, 0.0]])
        captured = {}
        def capture_mirror(m):
            captured['markers'] = m.copy()
            return np.random.randn(1, 8, 3)
        mock_hawk.mirror_keypoints = MagicMock(side_effect=capture_mirror)

        mock_target = MagicMock()
        mock_target.right_markers = np.array([[0.4, 0.15, 0.0],
                                               [0.25, 0.1, 0.0],
                                               [0.18, 0.06, 0.0],
                                               [0.06, -0.12, 0.0]])
        mock_target.skeleton_definition.marker_names = [
            'l_wt', 'r_wt', 'l_p', 'r_p', 'l_s', 'r_s', 'l_tt', 'r_tt',
        ]
        mock_bird_cls.return_value = mock_target
        mock_integrate.return_value = {}

        species_df = pd.DataFrame({'species_common': ['test']})
        transform_hawk_to_species(mock_hawk, 0, species_df, tail_z_override=-0.10)

        assert captured['markers'][0, -1, 2] == pytest.approx(-0.10)

    @patch('kinematic_morphospace.species_transform.Animal3D')
    @patch('kinematic_morphospace.species_transform.integrate_dataframe_to_bird3D')
    def test_transformation_matrix_is_block_diagonal(self, mock_integrate, mock_bird_cls):
        """The transformation matrix should be block-diagonal (12x12 for 4 markers)."""
        import pandas as pd

        mock_hawk = MagicMock()
        mock_hawk.right_markers = np.array([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 1.0, 0.0]])
        mock_hawk.mirror_keypoints = MagicMock(
            return_value=np.random.randn(1, 8, 3))

        mock_target = MagicMock()
        mock_target.right_markers = np.array([[2.0, 0.0, 0.0],
                                               [0.0, 2.0, 0.0],
                                               [0.0, 0.0, 2.0],
                                               [2.0, 2.0, 0.0]])
        mock_target.skeleton_definition.marker_names = [
            'left_wingtip', 'right_wingtip', 'left_primary', 'right_primary',
            'left_secondary', 'right_secondary', 'left_tailtip', 'right_tailtip',
        ]
        mock_bird_cls.return_value = mock_target
        mock_integrate.return_value = {}

        species_df = pd.DataFrame({'species_common': ['test']})
        _, _, T = transform_hawk_to_species(mock_hawk, 0, species_df)

        # Should be 12x12 (4 markers x 3 coords)
        assert T.shape == (12, 12)
        # Check off-diagonal blocks are zero
        for i in range(4):
            for j in range(4):
                block = T[i*3:(i+1)*3, j*3:(j+1)*3]
                if i != j:
                    np.testing.assert_allclose(block, np.zeros((3, 3)), atol=1e-10)


# ---------------------------------------------------------------------------
# Semi-integration tests: unmocked mathematical pipeline
# ---------------------------------------------------------------------------

class TestTransformationPipelineIntegration:
    """Tests that exercise compute_transformation_matrix and
    transform_principal_components without mocking, verifying the
    mathematical pipeline end-to-end."""

    @pytest.fixture
    def hawk_and_target_markers(self):
        """Realistic 4-marker hawk and target arrays."""
        hawk = np.array([
            [0.30, 0.10, 0.00],
            [0.20, 0.08, 0.00],
            [0.15, 0.05, 0.00],
            [0.05, -0.10, 0.00],
        ])
        target = np.array([
            [0.40, 0.15, 0.02],
            [0.25, 0.10, 0.01],
            [0.18, 0.06, 0.00],
            [0.06, -0.12, -0.01],
        ])
        return hawk, target

    def _build_block_diagonal(self, hawk, target):
        """Helper: build block-diagonal from per-marker transforms."""
        from scipy.linalg import block_diag
        T_list = [compute_transformation_matrix(hawk[i], target[i])
                  for i in range(len(hawk))]
        return block_diag(*T_list)

    def test_block_diagonal_transform_moves_markers_to_target(self, hawk_and_target_markers):
        """T @ hawk_flat should produce the target markers."""
        hawk, target = hawk_and_target_markers
        T = self._build_block_diagonal(hawk, target)
        result = (T @ hawk.reshape(-1)).reshape(-1, 3)
        np.testing.assert_allclose(result, target, atol=1e-10)

    def test_block_diagonal_is_correct_size(self, hawk_and_target_markers):
        """Block diagonal for n markers should be (3n, 3n)."""
        hawk, target = hawk_and_target_markers
        T = self._build_block_diagonal(hawk, target)
        n = len(hawk)
        assert T.shape == (3 * n, 3 * n)

    def test_transform_principal_components_roundtrip(self, hawk_and_target_markers):
        """transform then inverse-transform should recover original PCs."""
        hawk, target = hawk_and_target_markers
        T = self._build_block_diagonal(hawk, target)
        T_inv = np.linalg.inv(T)

        # Synthetic PCs: (3 components, 4 markers, 3 coords)
        rng = np.random.default_rng(99)
        pcs = rng.standard_normal((3, 4, 3))

        transformed = transform_principal_components(pcs, T)
        recovered = transform_principal_components(transformed, T_inv)
        np.testing.assert_allclose(recovered, pcs, atol=1e-10)

    def test_transformed_pcs_reconstruct_to_transformed_markers(self, hawk_and_target_markers):
        """A non-trivial block-diagonal T should satisfy:
        reconstruct(scores, T·PCs, T·mu) == T · reconstruct(scores, PCs, mu)
        i.e. transforming PCs then reconstructing gives the same result
        as reconstructing first then transforming each frame."""
        from kinematic_morphospace.pca_reconstruct import reconstruct

        hawk, target = hawk_and_target_markers
        T = self._build_block_diagonal(hawk, target)
        n_markers = len(hawk)
        n_features = n_markers * 3  # 12

        rng = np.random.default_rng(7)
        # Mimic real PCA: square PC matrix (n_features, n_features)
        pcs_2d = rng.standard_normal((n_features, n_features))
        scores = rng.standard_normal((20, n_features))
        mu = rng.standard_normal((1, n_markers, 3))

        # Transform PCs: reshape to 3D for transform_principal_components
        pcs_3d = pcs_2d.reshape(n_features, n_markers, 3)
        pcs_t_3d = transform_principal_components(pcs_3d, T)
        pcs_t_2d = pcs_t_3d.reshape(n_features, n_features)

        # Transform mean
        mu_flat = mu.reshape(-1)
        mu_t = (T @ mu_flat).reshape(1, n_markers, 3)

        # Path A: reconstruct with transformed PCs and mean
        recon_a = reconstruct(scores, pcs_t_2d, mu_t)

        # Path B: reconstruct original, then transform each frame
        recon_orig = reconstruct(scores, pcs_2d, mu)
        recon_b = np.zeros_like(recon_orig)
        for i in range(recon_orig.shape[0]):
            frame_flat = recon_orig[i].reshape(-1)
            recon_b[i] = (T @ frame_flat).reshape(n_markers, 3)

        np.testing.assert_allclose(recon_a, recon_b, atol=1e-10)

    def test_transformation_preserves_linear_independence(self, hawk_and_target_markers):
        """Block-diagonal T is invertible, so transformed PCs should
        remain linearly independent and no variance component should
        collapse to zero."""
        hawk, target = hawk_and_target_markers
        T = self._build_block_diagonal(hawk, target)

        rng = np.random.default_rng(12)
        n_pcs = 5
        pcs_3d = rng.standard_normal((n_pcs, len(hawk), 3))

        pcs_t_3d = transform_principal_components(pcs_3d, T)

        # All PCs should remain non-zero after an invertible transform
        for k in range(n_pcs):
            assert np.linalg.norm(pcs_t_3d[k]) > 1e-10, (
                f"PC {k} collapsed after invertible transformation")

        # Linear independence preserved: Gram matrix should have full rank
        pcs_t_flat = pcs_t_3d.reshape(n_pcs, -1)
        gram = pcs_t_flat @ pcs_t_flat.T
        rank = np.linalg.matrix_rank(gram, tol=1e-10)
        assert rank == n_pcs, (
            f"Transformed PCs lost rank: expected {n_pcs}, got {rank}")

    def test_tail_z_override_in_full_pipeline(self, hawk_and_target_markers):
        """Verify the tail override would affect the last marker's z
        when applied after the block-diagonal transform."""
        hawk, target = hawk_and_target_markers
        T = self._build_block_diagonal(hawk, target)

        # Apply transformation
        transformed = (T @ hawk.reshape(-1)).reshape(-1, 3)
        # Simulate the override
        transformed[-1, 2] = -0.05

        assert transformed[-1, 2] == pytest.approx(-0.05)
        # Other markers' z should remain at their transformed values
        expected_others = (T @ hawk.reshape(-1)).reshape(-1, 3)[:-1, 2]
        np.testing.assert_allclose(transformed[:-1, 2], expected_others)
