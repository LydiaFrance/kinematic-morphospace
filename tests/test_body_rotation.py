"""Tests for kinematic_morphospace.preprocessing.body_rotation."""
from __future__ import annotations

import numpy as np
import pytest

from kinematic_morphospace.preprocessing.body_rotation import (
    apply_rotation,
    build_body_frame,
    build_rotation_matrices,
    compute_pitch_angle,
    compute_yaw_angle,
    extract_body_angles,
    rotate_to_body_frame,
)


class TestComputePitchAngle:
    """Tests for compute_pitch_angle (vectorized find_pitchangle.m)."""

    def test_straight_back(self):
        # Vector pointing straight backward [0, -1, 0] -> pitch = 0
        v = np.array([[0.0, -1.0, 0.0]])
        result = compute_pitch_angle(v)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_straight_down(self):
        # Vector pointing straight down [0, 0, -1] -> pitch = 90
        v = np.array([[0.0, 0.0, -1.0]])
        result = compute_pitch_angle(v)
        np.testing.assert_allclose(result, [90.0], atol=1e-10)

    def test_straight_up(self):
        # Vector pointing straight up [0, 0, 1] -> pitch = -90
        v = np.array([[0.0, 0.0, 1.0]])
        result = compute_pitch_angle(v)
        np.testing.assert_allclose(result, [-90.0], atol=1e-10)

    def test_x_component_ignored(self):
        # X component should be zeroed
        v1 = np.array([[0.0, -1.0, 0.0]])
        v2 = np.array([[5.0, -1.0, 0.0]])
        np.testing.assert_allclose(
            compute_pitch_angle(v1),
            compute_pitch_angle(v2),
            atol=1e-10,
        )

    def test_batch(self):
        v = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, -1.0],
        ])
        result = compute_pitch_angle(v)
        assert result.shape == (3,)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1], 90.0, atol=1e-10)
        np.testing.assert_allclose(result[2], 45.0, atol=1e-10)

    def test_1d_input(self):
        v = np.array([0.0, -1.0, 0.0])
        result = compute_pitch_angle(v)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)


class TestComputeYawAngle:
    """Tests for compute_yaw_angle (vectorized find_yawangle.m)."""

    def test_forward_is_zero(self):
        # Vector pointing along +Y -> yaw = 0
        v = np.array([[0.0, 1.0, 0.0]])
        result = compute_yaw_angle(v)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_sideways_is_90(self):
        # Vector pointing along +X -> yaw = 90
        v = np.array([[1.0, 0.0, 0.0]])
        result = compute_yaw_angle(v)
        np.testing.assert_allclose(result, [90.0], atol=1e-10)

    def test_backward_is_180(self):
        # Vector pointing along -Y -> yaw = 180
        v = np.array([[0.0, -1.0, 0.0]])
        result = compute_yaw_angle(v)
        np.testing.assert_allclose(result, [180.0], atol=1e-10)

    def test_z_component_ignored(self):
        v1 = np.array([[0.0, 1.0, 0.0]])
        v2 = np.array([[0.0, 1.0, 5.0]])
        np.testing.assert_allclose(
            compute_yaw_angle(v1),
            compute_yaw_angle(v2),
            atol=1e-10,
        )


class TestBuildRotationMatrices:
    """Tests for build_rotation_matrices."""

    def test_identity_at_zero(self):
        R = build_rotation_matrices(np.array([0.0]), axis="x")
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_x_90_degrees(self):
        R = build_rotation_matrices(np.array([90.0]), axis="x")
        # Rx(90) should rotate [0,1,0] to [0,0,1]
        v = np.array([0.0, 1.0, 0.0])
        rotated = R[0] @ v
        np.testing.assert_allclose(rotated, [0.0, 0.0, 1.0], atol=1e-10)

    def test_batch_shape(self):
        R = build_rotation_matrices(np.array([0.0, 45.0, 90.0]), axis="y")
        assert R.shape == (3, 3, 3)

    def test_orthogonality(self):
        angles = np.random.default_rng(42).uniform(-180, 180, size=100)
        for axis in ("x", "y", "z"):
            R = build_rotation_matrices(angles, axis=axis)
            for i in range(len(angles)):
                eye = R[i] @ R[i].T
                np.testing.assert_allclose(eye, np.eye(3), atol=1e-10)

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="axis must be"):
            build_rotation_matrices(np.array([0.0]), axis="w")


class TestApplyRotation:
    """Tests for apply_rotation (vectorized matrix-vector product)."""

    def test_identity(self):
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        R = np.stack([np.eye(3)] * 2)
        result = apply_rotation(xyz, R)
        np.testing.assert_allclose(result, xyz, atol=1e-10)

    def test_matches_matlab_convention(self):
        # MATLAB: xyz * rotmat (row vector * matrix)
        # Rx(90) applied to [0,1,0] -> [0,0,-1] in row-vector convention
        angle = 90.0
        R = build_rotation_matrices(np.array([angle]), axis="x")
        xyz = np.array([[0.0, 1.0, 0.0]])
        result = apply_rotation(xyz, R)
        np.testing.assert_allclose(result, [[0.0, 0.0, -1.0]], atol=1e-10)


class TestBuildBodyFrame:
    """Tests for build_body_frame."""

    def test_forward_vector(self):
        # Tail pointing backward along -Y
        tail = np.array([[0.0, -1.0, 0.0]])
        body, side, up = build_body_frame(tail)

        # Body axis should be normalized -Y
        np.testing.assert_allclose(body, [[0.0, -1.0, 0.0]], atol=1e-10)

        # All should be unit vectors
        np.testing.assert_allclose(np.linalg.norm(body, axis=1), [1.0])
        np.testing.assert_allclose(np.linalg.norm(side, axis=1), [1.0])
        np.testing.assert_allclose(np.linalg.norm(up, axis=1), [1.0])

    def test_orthogonality(self):
        rng = np.random.default_rng(42)
        tail = rng.standard_normal((50, 3))
        # Avoid vectors nearly parallel to z-axis
        tail[:, 2] = tail[:, 2] * 0.1

        body, side, up = build_body_frame(tail)

        # All three axes should be mutually orthogonal
        for i in range(len(tail)):
            np.testing.assert_allclose(
                np.dot(body[i], side[i]), 0.0, atol=1e-10
            )
            np.testing.assert_allclose(
                np.dot(body[i], up[i]), 0.0, atol=1e-10
            )
            np.testing.assert_allclose(
                np.dot(side[i], up[i]), 0.0, atol=1e-10
            )

    def test_determinant_positive(self):
        # Body frame should be right-handed (det = +1)
        tail = np.array([[0.0, -1.0, -0.3], [0.2, -0.8, 0.1]])
        body, side, up = build_body_frame(tail)

        for i in range(len(tail)):
            R = np.stack([side[i], body[i], up[i]])
            det = np.linalg.det(R)
            np.testing.assert_allclose(abs(det), 1.0, atol=1e-10)


class TestRotateToBodyFrame:
    """Tests for rotate_to_body_frame."""

    def test_identity_frame(self):
        # Body frame aligned with global frame
        n = 5
        xyz = np.random.default_rng(42).standard_normal((n, 3))
        side = np.tile([1.0, 0.0, 0.0], (n, 1))
        body = np.tile([0.0, 1.0, 0.0], (n, 1))
        up = np.tile([0.0, 0.0, 1.0], (n, 1))

        result = rotate_to_body_frame(xyz, side, body, up)
        np.testing.assert_allclose(result, xyz, atol=1e-10)

    def test_rotation_preserves_norm(self):
        rng = np.random.default_rng(42)
        tail = rng.standard_normal((20, 3))
        tail[:, 2] *= 0.1
        body, side, up = build_body_frame(tail)

        xyz = rng.standard_normal((20, 3))
        rotated = rotate_to_body_frame(xyz, side, body, up)

        original_norms = np.linalg.norm(xyz, axis=1)
        rotated_norms = np.linalg.norm(rotated, axis=1)
        np.testing.assert_allclose(rotated_norms, original_norms, atol=1e-10)


class TestExtractBodyAngles:
    """Tests for extract_body_angles."""

    def test_identity_frame_zero_angles(self):
        n = 5
        body = np.tile([0.0, -1.0, 0.0], (n, 1))
        side = np.tile([1.0, 0.0, 0.0], (n, 1))
        up = np.tile([0.0, 0.0, 1.0], (n, 1))

        pitch, yaw, roll = extract_body_angles(body, side, up)
        # Pitch of [0, -1, 0] should be 0
        np.testing.assert_allclose(pitch, 0.0, atol=1e-10)

    def test_returns_three_arrays(self):
        rng = np.random.default_rng(42)
        tail = rng.standard_normal((10, 3))
        tail[:, 2] *= 0.1
        body, side, up = build_body_frame(tail)

        pitch, yaw, roll = extract_body_angles(body, side, up)
        assert pitch.shape == (10,)
        assert yaw.shape == (10,)
        assert roll.shape == (10,)

    def test_pitch_in_expected_range(self):
        # After wrapping corrections, pitch should be in [-90, 360] range
        rng = np.random.default_rng(42)
        tail = rng.standard_normal((100, 3))
        tail[:, 2] *= 0.3
        body, side, up = build_body_frame(tail)

        pitch, _, _ = extract_body_angles(body, side, up)
        assert np.all(pitch >= -100)
        assert np.all(pitch <= 360)
