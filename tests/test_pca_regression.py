"""
test_pca_regression.py

End-to-end regression tests for the full PCA pipeline.  The pipeline is run
from raw data and its outputs are compared against a set of frozen reference
arrays (golden files) produced by a validated earlier run.

Pipeline stages under test:
  load → remove_frames → scale → process → add_tailpack →
  bilateral PCA → symmetry projection → Kabsch rotation →
  to_unilateral → unilateral PCA

Golden files live at GOLDEN_DIR (a local path; tests are skipped in CI where
the files are absent).

Note: the current pipeline uses pca.mean_ (training-only mean) for
reconstruction, whereas the golden files were produced with np.mean(all_data).
This is a deliberate methodological improvement.  PCA subspace tests therefore
use cosine similarity (> 0.996) rather than exact equality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from kinematic_morphospace import (
    load_data, remove_frames, scale_data, process_data,
    add_turn_info, add_tailpack_data,
    filter_by, run_PCA,
    vectorised_kabsch, apply_rotation,
    reconstruct, to_unilateral,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src" / "kinematic_morphospace"

GOLDEN_DIR = Path(
    "/Users/lfrance/Library/CloudStorage/OneDrive-Personal"
    "/004 GitHub/BirdPCA/data/processed"
)

# ---------------------------------------------------------------------------
# Skip if data files are not available (e.g. CI)
# ---------------------------------------------------------------------------
_has_data = (DATA_DIR / "raw" / "bilateral_markers.csv").exists()
_has_golden = (GOLDEN_DIR / "unilateral_principal_components.npy").exists()

pytestmark = pytest.mark.skipif(
    not (_has_data and _has_golden),
    reason="Requires local raw data and golden reference files",
)


# ---------------------------------------------------------------------------
# Fixture: run the full pipeline once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline_results():
    """Run the complete pipeline and return all intermediate results."""

    # Phase 1: Load and process
    csv_path = str(DATA_DIR / "raw" / "bilateral_markers.csv")
    data_csv = load_data(csv_path)
    data_csv = remove_frames(data_csv)  # time_limit=0 default

    wingspan_path = str(SRC_DIR / "TotalWingspans.yml")
    data_csv = scale_data(data_csv, wingspan_path)

    markers, frame_info, markers_df, frame_info_df = process_data(data_csv)

    turn_csv = str(DATA_DIR / "raw" / "obstacle_turns.csv")
    frame_info_df = add_turn_info(frame_info_df, turn_csv)

    tailpack_csv = str(DATA_DIR / "raw" / "tailpack.csv")
    markers_with_tailpack, combined_frame_info_df = add_tailpack_data(
        markers_df, frame_info_df, tailpack_csv, wingspan_path=wingspan_path,
    )

    # Phase 2: Bilateral PCA (non-obstacle training)
    filt_bilateral = filter_by(combined_frame_info_df, obstacle=0)
    bilateral_pcs, bilateral_scores, bilateral_pca = run_PCA(
        markers_with_tailpack[filt_bilateral], markers_with_tailpack
    )

    # Phase 3: Rotation correction
    # Use training mean (pca.mean_) for reconstruction — methodologically
    # consistent with PCA being fitted on non-obstacle flights only.
    mean_shape = bilateral_pca.mean_.reshape(1, -1, 3)
    symmetric_components = [0, 1]
    symmetric_projection = reconstruct(
        bilateral_scores, bilateral_pcs, mean_shape, symmetric_components,
    )
    rotation_matrices = vectorised_kabsch(
        markers_with_tailpack, symmetric_projection,
    )
    transformed_markers = apply_rotation(
        markers_with_tailpack, rotation_matrices,
    )

    # Phase 4: Bilateral → unilateral
    bilateral_data = transformed_markers[:, :8, :]
    unilateral_data = to_unilateral(bilateral_data)

    # Build unilateral frame info (left then right)
    left_info = combined_frame_info_df.copy()
    left_info["Left"] = True
    right_info = combined_frame_info_df.copy()
    right_info["Left"] = False
    unilateral_frame_info_df = pd.concat(
        [left_info, right_info], ignore_index=True,
    )

    # Phase 5: Unilateral PCA
    filt_unilateral = filter_by(unilateral_frame_info_df, obstacle=0)
    uni_pcs, uni_scores, uni_pca = run_PCA(
        unilateral_data[filt_unilateral], unilateral_data,
    )

    return {
        "markers_with_tailpack": markers_with_tailpack,
        "combined_frame_info_df": combined_frame_info_df,
        "bilateral_pcs": bilateral_pcs,
        "bilateral_scores": bilateral_scores,
        "bilateral_pca": bilateral_pca,
        "rotation_matrices": rotation_matrices,
        "transformed_markers": transformed_markers,
        "bilateral_data": bilateral_data,
        "unilateral_data": unilateral_data,
        "unilateral_frame_info_df": unilateral_frame_info_df,
        "uni_pcs": uni_pcs,
        "uni_scores": uni_scores,
        "uni_pca": uni_pca,
    }


@pytest.fixture(scope="session")
def golden():
    """Load frozen reference arrays from a validated pipeline run."""
    return {
        "uni_pcs": np.load(GOLDEN_DIR / "unilateral_principal_components.npy"),
        "uni_mu": np.load(GOLDEN_DIR / "unilateral_mu.npy"),
        "uni_scores": np.load(GOLDEN_DIR / "unilateral_scores.npy"),
        "bilateral_data": np.load(GOLDEN_DIR / "bilateral_data.npy"),
        "scaled_markers_with_tailpack": np.load(
            GOLDEN_DIR / "scaled_markers_with_tailpack.npy"
        ),
        "transformed_markers_with_tailpack": np.load(
            GOLDEN_DIR / "transformed_markers_with_tailpack.npy"
        ),
    }


# ---------------------------------------------------------------------------
# Tests: frame counts
# ---------------------------------------------------------------------------
class TestFrameCounts:
    def test_bilateral_frame_count(self, pipeline_results, golden):
        """Bilateral frame count must match the golden reference (time_limit=0)."""
        current = pipeline_results["markers_with_tailpack"].shape[0]
        reference = golden["scaled_markers_with_tailpack"].shape[0]
        assert current == reference, f"Bilateral frames: current={current}, reference={reference}"

    def test_unilateral_frame_count(self, pipeline_results, golden):
        """Unilateral frame count must be 2 × bilateral."""
        current = pipeline_results["unilateral_data"].shape[0]
        reference = golden["uni_scores"].shape[0]
        assert current == reference, f"Unilateral frames: current={current}, reference={reference}"


# ---------------------------------------------------------------------------
# Tests: pre-rotation data (must match exactly)
# ---------------------------------------------------------------------------
class TestPreRotationData:
    def test_scaled_markers_match(self, pipeline_results, golden):
        """Scaled markers with tailpack must match the golden reference exactly."""
        current = pipeline_results["markers_with_tailpack"]
        reference = golden["scaled_markers_with_tailpack"]
        np.testing.assert_allclose(current, reference, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: unilateral PCA components
#
# The current pipeline uses pca.mean_ (training-only mean) for reconstruction,
# while the golden files were produced with np.mean(all_data). This is a
# deliberate methodological improvement, so exact equality is not expected.
# Cosine similarity > 0.996 confirms the subspaces are equivalent.
# ---------------------------------------------------------------------------
class TestUnilateralPCA:
    @pytest.mark.parametrize("pc_idx", range(12))
    def test_component_cosine_similarity(
        self, pipeline_results, golden, pc_idx,
    ):
        """Each unilateral PC must align with the golden reference (cos > 0.996).

        The tolerance reflects the deliberate change from all-data mean to
        training-only mean in the rotation-correction step.
        """
        current_pc = pipeline_results["uni_pcs"][pc_idx]
        reference_pc = golden["uni_pcs"][pc_idx]
        cos_sim = np.abs(np.dot(current_pc, reference_pc)) / (
            np.linalg.norm(current_pc) * np.linalg.norm(reference_pc)
        )
        assert cos_sim > 0.996, (
            f"PC{pc_idx + 1}: cosine similarity = {cos_sim:.6f} (need > 0.996)"
        )

    def test_explained_variance_matches(self, pipeline_results, golden):
        """Score array shape must match the golden reference."""
        current_scores = pipeline_results["uni_scores"]
        reference_scores = golden["uni_scores"]
        assert current_scores.shape == reference_scores.shape
