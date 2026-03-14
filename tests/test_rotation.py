"""
test_rotation.py

Tests for the rotation module: assess_symmetry(), vectorised_kabsch(),
extract_euler_angles_from_matrices(), apply_rotation(),
undo_body_pitch_rotation(), undo_body_rotation().

Covers S05 (Projection and Rotation Analysis) and S07 (Comparing Before
and After Rotation) from the supplementary spec.
"""

import logging
import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from kinematic_morphospace.rotation import (
    assess_symmetry,
    vectorised_kabsch,
    extract_euler_angles_from_matrices,
    apply_rotation,
    undo_body_pitch_rotation,
    undo_body_rotation,
)

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def symmetric_pc():
    """A perfectly symmetric PC: right marker = left marker mirrored in x.

    8 markers arranged as pairs (0,1), (2,3), (4,5), (6,7).
    Each marker has [x, y, z]. Right marker has x negated.
    """
    pc = np.zeros((8, 3))
    for i in range(0, 8, 2):
        pc[i] = [1.0 + i, 2.0, 3.0]       # left
        pc[i + 1] = [-(1.0 + i), 2.0, 3.0]  # right (mirrored in x)
    return pc


@pytest.fixture
def asymmetric_pc():
    """An asymmetric PC where left != mirrored right."""
    pc = np.zeros((8, 3))
    for i in range(0, 8, 2):
        pc[i] = [1.0, 2.0, 3.0]       # left
        pc[i + 1] = [0.5, 2.0, 3.0]   # right (NOT mirrored in x)
    return pc


@pytest.fixture
def synthetic_markers_3d():
    """Batch of marker data: (n_instances, n_markers, 3)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 8, 3))


@pytest.fixture
def identity_rotations():
    """Batch of identity rotation matrices."""
    n = 50
    return np.tile(np.eye(3), (n, 1, 1))


# -- assess_symmetry tests --

class TestAssessSymmetry:
    def test_perfect_symmetry_returns_zero(self, symmetric_pc):
        """Perfectly mirrored left-right markers should have symmetry = 0."""
        score = assess_symmetry(symmetric_pc, axis='x', nMarkers=8)
        assert score == pytest.approx(0.0)

    def test_asymmetric_returns_positive(self, asymmetric_pc):
        """Asymmetric markers should have symmetry > 0."""
        score = assess_symmetry(asymmetric_pc, axis='x', nMarkers=8)
        assert score > 0

    def test_axis_y(self):
        """Symmetry around y-axis: right marker y negated."""
        pc = np.array([
            [1.0, 2.0, 3.0],
            [1.0, -2.0, 3.0],  # mirrored in y
            [4.0, 5.0, 6.0],
            [4.0, -5.0, 6.0],  # mirrored in y
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        score = assess_symmetry(pc, axis='y', nMarkers=8)
        assert score == pytest.approx(0.0)

    def test_axis_z(self):
        """Symmetry around z-axis."""
        pc = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, -3.0],  # mirrored in z
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        score = assess_symmetry(pc, axis='z', nMarkers=8)
        assert score == pytest.approx(0.0)

    def test_invalid_axis_raises(self, symmetric_pc):
        with pytest.raises(ValueError, match="Invalid axis"):
            assess_symmetry(symmetric_pc, axis='w')

    def test_exact_numerical_value(self):
        """Hand-calculated sum-of-squared-differences for 4 markers.

        Pair 0: left=[1,2,3], right=[0.5,2,3] → mirrored=[-0.5,2,3]
                diff=[1.5,0,0] → sq_sum=2.25
        Pair 1: left=[1,1,1], right=[-1,2,1] → mirrored=[1,2,1]
                diff=[0,-1,0] → sq_sum=1.0
        Total = 3.25
        """
        pc = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 2.0, 3.0],
            [1.0, 1.0, 1.0],
            [-1.0, 2.0, 1.0],
        ])
        score = assess_symmetry(pc, axis='x', nMarkers=4)
        assert score == pytest.approx(3.25)

    def test_custom_nMarkers(self):
        """Works with 4 markers (2 pairs)."""
        pc = np.array([
            [1.0, 2.0, 3.0],
            [-1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-4.0, 5.0, 6.0],
        ])
        score = assess_symmetry(pc, axis='x', nMarkers=4)
        assert score == pytest.approx(0.0)


# -- vectorised_kabsch tests --

class TestVectorisedKabsch:
    def test_identity_when_P_equals_Q(self, synthetic_markers_3d):
        """When P == Q, the optimal rotation should be identity."""
        P = synthetic_markers_3d
        Q = synthetic_markers_3d.copy()
        rotations = vectorised_kabsch(P, Q)
        expected = np.tile(np.eye(3), (P.shape[0], 1, 1))
        np.testing.assert_allclose(rotations, expected, atol=1e-10)

    def test_finds_known_rotation(self):
        """Apply a known rotation to P, kabsch should recover it."""
        rng = np.random.default_rng(123)
        P = rng.standard_normal((20, 8, 3))

        # Known rotation: 45 degrees around z-axis
        angle = np.radians(45)
        R_known = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])

        # Q = P @ R_known for each instance
        Q = np.einsum('nij,jk->nik', P, R_known)

        rotations = vectorised_kabsch(P, Q)

        # Each rotation should be close to R_known
        for i in range(20):
            np.testing.assert_allclose(rotations[i], R_known, atol=1e-10)

    def test_result_is_proper_rotation(self, synthetic_markers_3d):
        """Rotation matrices should have det=1 and R^T @ R = I."""
        rng = np.random.default_rng(99)
        Q = rng.standard_normal(synthetic_markers_3d.shape)
        rotations = vectorised_kabsch(synthetic_markers_3d, Q)

        for i in range(rotations.shape[0]):
            # Orthogonal
            np.testing.assert_allclose(
                rotations[i].T @ rotations[i], np.eye(3), atol=1e-10
            )
            # Proper rotation (det = +1)
            det = np.linalg.det(rotations[i])
            assert det == pytest.approx(1.0, abs=1e-10)

    def test_batch_size_mismatch_raises(self):
        P = np.zeros((10, 4, 3))
        Q = np.zeros((5, 4, 3))
        with pytest.raises(ValueError, match="same batch size"):
            vectorised_kabsch(P, Q)

    def test_finds_rotation_with_offset_data(self):
        """Markers offset from origin should still recover the correct rotation.

        Without centroid centering, the Kabsch SVD operates on un-centred
        cross-covariance and returns incorrect rotations for offset data.
        """
        rng = np.random.default_rng(456)
        P = rng.standard_normal((15, 8, 3))

        # Offset markers far from origin
        offset = np.array([10.0, 20.0, 30.0])
        P_offset = P + offset

        # Known rotation: 60 degrees around z-axis
        angle = np.radians(60)
        R_known = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])

        # Rotate around the centroid (offset preserved)
        Q_offset = np.einsum('nij,jk->nik', P_offset - offset, R_known) + offset

        rotations = vectorised_kabsch(P_offset, Q_offset, centre=True)

        for i in range(15):
            np.testing.assert_allclose(rotations[i], R_known, atol=1e-10)

    def test_shape_mismatch_raises(self):
        P = np.zeros((10, 4, 3))
        Q = np.zeros((10, 8, 3))
        with pytest.raises(ValueError, match="same shape"):
            vectorised_kabsch(P, Q)


# -- extract_euler_angles_from_matrices tests --

class TestExtractEulerAngles:
    def test_identity_gives_zero_angles(self):
        identity = np.eye(3).reshape(1, 3, 3)
        angles = extract_euler_angles_from_matrices(identity)
        np.testing.assert_allclose(angles, 0.0, atol=1e-10)

    def test_known_rotation_angles(self):
        """90-degree rotation around z-axis should give [0, 0, 90]."""
        angle = np.radians(90)
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]).reshape(1, 3, 3)
        angles = extract_euler_angles_from_matrices(rot, sequence='xyz')
        # For xyz sequence, a z-rotation of 90 deg
        assert angles[0, 2] == pytest.approx(90.0, abs=0.1)

    def test_batch_of_rotations(self, synthetic_markers_3d):
        """Should return one set of angles per rotation matrix."""
        n = 10
        # Create random rotation matrices
        scipy_rots = R.random(n, random_state=42)
        matrices = scipy_rots.as_matrix()
        angles = extract_euler_angles_from_matrices(matrices, sequence='xyz')
        assert angles.shape == (n, 3)

    def test_custom_sequence(self):
        """Different Euler sequence should change the decomposition."""
        scipy_rot = R.from_euler('xyz', [30, 45, 60], degrees=True)
        matrix = scipy_rot.as_matrix().reshape(1, 3, 3)

        angles_xyz = extract_euler_angles_from_matrices(matrix, sequence='xyz')
        angles_zyx = extract_euler_angles_from_matrices(matrix, sequence='zyx')

        # Different sequences produce different angle decompositions
        assert not np.allclose(angles_xyz, angles_zyx)


# -- apply_rotation tests --

class TestApplyRotation:
    def test_identity_rotation_preserves_data(self, synthetic_markers_3d, identity_rotations):
        result = apply_rotation(synthetic_markers_3d, identity_rotations)
        np.testing.assert_allclose(result, synthetic_markers_3d, atol=1e-12)

    def test_known_rotation_applied(self):
        """Apply 90-degree z-rotation: [1,0,0] -> [0,1,0]."""
        P = np.array([[[1.0, 0.0, 0.0]]])  # (1, 1, 3)
        angle = np.radians(90)
        rot = np.array([[
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]])  # (1, 3, 3) — right-multiply, so transpose convention
        result = apply_rotation(P, rot)
        np.testing.assert_allclose(result[0, 0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_output_shape_matches_input(self, synthetic_markers_3d, identity_rotations):
        result = apply_rotation(synthetic_markers_3d, identity_rotations)
        assert result.shape == synthetic_markers_3d.shape

    def test_kabsch_then_apply_roundtrip(self):
        """Apply known rotation, find it with kabsch, apply inverse → recover original."""
        rng = np.random.default_rng(77)
        P = rng.standard_normal((30, 8, 3))

        # Known rotation
        angle = np.radians(30)
        R_known = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        Q = np.einsum('nij,jk->nik', P, R_known)

        # Find rotation with kabsch
        rotations = vectorised_kabsch(P, Q)

        # Apply rotation to P → should get Q
        P_rotated = apply_rotation(P, rotations)
        np.testing.assert_allclose(P_rotated, Q, atol=1e-10)


# -- undo_body_pitch_rotation tests --

class TestUndoBodyPitchRotation:
    def test_zero_pitch_preserves_markers(self, synthetic_markers_3d):
        """Zero pitch angle should return markers unchanged."""
        body_pitch = np.zeros(synthetic_markers_3d.shape[0])
        result = undo_body_pitch_rotation(synthetic_markers_3d, body_pitch)
        np.testing.assert_allclose(result, synthetic_markers_3d, atol=1e-12)

    def test_roundtrip(self, synthetic_markers_3d):
        """Applying pitch then undoing it should recover original markers."""
        pitch_angles = np.full(synthetic_markers_3d.shape[0], 30.0)
        # Apply pitch
        rotated = undo_body_pitch_rotation(synthetic_markers_3d, pitch_angles)
        # Undo by applying negative pitch
        recovered = undo_body_pitch_rotation(rotated, -pitch_angles)
        np.testing.assert_allclose(recovered, synthetic_markers_3d, atol=1e-10)

    def test_known_pitch_rotation(self):
        """90-degree pitch: y->z, z->-y (rotation about x-axis)."""
        markers = np.array([[[0.0, 1.0, 0.0]]])  # (1, 1, 3)
        pitch = np.array([90.0])
        result = undo_body_pitch_rotation(markers, pitch)
        # Rotation about x by 90 deg: [0,1,0] -> [0,0,1]
        np.testing.assert_allclose(result[0, 0], [0.0, 0.0, 1.0], atol=1e-10)


# -- undo_body_rotation tests --

class TestUndoBodyRotation:
    def test_zero_angle_preserves_markers(self, synthetic_markers_3d):
        angles = np.zeros(synthetic_markers_3d.shape[0])
        for axis in ['x', 'y', 'z']:
            result = undo_body_rotation(synthetic_markers_3d, angles, which_axis=axis)
            np.testing.assert_allclose(result, synthetic_markers_3d, atol=1e-12)

    def test_invalid_axis_raises(self, synthetic_markers_3d):
        angles = np.zeros(synthetic_markers_3d.shape[0])
        with pytest.raises(ValueError, match="Invalid axis"):
            undo_body_rotation(synthetic_markers_3d, angles, which_axis='w')

    def test_roundtrip_each_axis(self, synthetic_markers_3d):
        """Apply then reverse rotation for each axis."""
        angles = np.full(synthetic_markers_3d.shape[0], 45.0)
        for axis in ['x', 'y', 'z']:
            rotated = undo_body_rotation(synthetic_markers_3d, angles, which_axis=axis)
            recovered = undo_body_rotation(rotated, -angles, which_axis=axis)
            np.testing.assert_allclose(
                recovered, synthetic_markers_3d, atol=1e-10,
                err_msg=f"Roundtrip failed for axis '{axis}'"
            )

    def test_z_axis_matches_pitch_rotation(self, synthetic_markers_3d):
        """undo_body_rotation with axis='z' should match undo_body_pitch_rotation
        since both apply rotation about x-axis (the 'z' case in undo_body_rotation
        uses the same matrix as undo_body_pitch_rotation)."""
        angles = np.full(synthetic_markers_3d.shape[0], 25.0)
        result_z = undo_body_rotation(synthetic_markers_3d, angles, which_axis='z')
        result_pitch = undo_body_pitch_rotation(synthetic_markers_3d, angles)
        np.testing.assert_allclose(result_z, result_pitch, atol=1e-12)

    def test_x_axis_known_rotation(self):
        """Rotation about y-axis (which_axis='x'): [1,0,0] -> [cos,0,-sin]."""
        markers = np.array([[[1.0, 0.0, 0.0]]])
        angle = np.array([90.0])
        result = undo_body_rotation(markers, angle, which_axis='x')
        # Ry(90): [1,0,0] -> [0,0,-1]
        np.testing.assert_allclose(result[0, 0], [0.0, 0.0, -1.0], atol=1e-10)

    def test_y_axis_known_rotation(self):
        """Rotation about z-axis (which_axis='y'): [1,0,0] -> [cos,sin,0].

        Matrix is [[cos,-sin,0],[sin,cos,0],[0,0,1]].
        Code applies markers @ R.T, so [1,0,0] @ R.T = [cos,sin,0].
        At 90 deg: [0, 1, 0].
        """
        markers = np.array([[[1.0, 0.0, 0.0]]])
        angle = np.array([90.0])
        result = undo_body_rotation(markers, angle, which_axis='y')
        np.testing.assert_allclose(result[0, 0], [0.0, 1.0, 0.0], atol=1e-10)


# -- Integration: rotation pipeline on real data --

class TestRotationPipeline:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_kabsch_rotation_correction_pipeline(self, sample_bilateraldata_path):
        """End-to-end test: load bilateral data, run PCA, find symmetric
        components, reconstruct, find rotation, apply it.
        Tests S05 pipeline from the supplementary spec."""
        from kinematic_morphospace import load_data, process_data
        from kinematic_morphospace.pca_core import run_PCA
        from kinematic_morphospace.pca_reconstruct import reconstruct

        data_csv = load_data(sample_bilateraldata_path)
        markers, _, _, _ = process_data(data_csv)
        n_frames = markers.shape[0]

        # Run PCA on bilateral data
        components, scores, pca_obj = run_PCA(markers)
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)

        # Symmetry check on first 2 components (reshaped to markers)
        n_markers = markers.shape[1]
        for pc_idx in range(min(2, components.shape[0])):
            pc_reshaped = components[pc_idx].reshape(n_markers, 3)
            score = assess_symmetry(pc_reshaped, axis='x', nMarkers=n_markers)
            logger.info(f"PC{pc_idx} symmetry score: {score:.6f}")
            # Symmetry scores should be finite
            assert np.isfinite(score)

        # Reconstruct using first 2 components (symmetric projection)
        reconstructed = reconstruct(scores, components, mu, components_list=[0, 1])
        assert reconstructed.shape == markers.shape

        # Find rotation matrices between original and symmetric projection
        rotation_matrices = vectorised_kabsch(
            reconstructed,
            markers.reshape(n_frames, n_markers, 3)
        )
        assert rotation_matrices.shape == (n_frames, 3, 3)

        # Apply rotation to original data
        corrected = apply_rotation(
            markers.reshape(n_frames, n_markers, 3),
            rotation_matrices
        )
        assert corrected.shape == markers.shape

        # Extract Euler angles for inspection
        angles = extract_euler_angles_from_matrices(rotation_matrices)
        assert angles.shape == (n_frames, 3)
        assert np.all(np.isfinite(angles))
