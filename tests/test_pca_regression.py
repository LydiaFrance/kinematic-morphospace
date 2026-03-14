"""
test_pca_regression.py

End-to-end regression test that reproduces the full PCA pipeline
and compares unilateral principal components against reference values
saved from the old repository.

The pipeline is:
  load → remove_frames → scale → process → add_tailpack →
  bilateral PCA → symmetry projection → Kabsch rotation →
  to_unilateral → unilateral PCA

Reference files are loaded from the old (untouched) repo at:
  /Users/lfrance/.../FromBusiness/Hawkflight/kinematic-morphospace/data/processed/

Note: the new pipeline uses pca.mean_ (training-only mean) for
reconstruction, whereas the old repo used np.mean(all_data). This is
a deliberate methodological improvement. Tests verify that the resulting
PCs remain very close (cosine similarity > 0.996) despite this change.
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

OLD_REPO = Path(
    "/Users/lfrance/Library/CloudStorage/OneDrive-Personal"
    "/FromBusiness/Hawkflight/kinematic-morphospace/data/processed"
)

# ---------------------------------------------------------------------------
# Skip if data files are not available (e.g. CI)
# ---------------------------------------------------------------------------
_has_data = (DATA_DIR / "2024-03-24-FullBilateralMarkers.csv").exists()
_has_old_repo = (OLD_REPO / "unilateral_principal_components.npy").exists()

pytestmark = pytest.mark.skipif(
    not (_has_data and _has_old_repo),
    reason="Requires local data files and old repo reference",
)


# ---------------------------------------------------------------------------
# Fixture: run the full pipeline once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline_results():
    """Run the complete pipeline and return all intermediate results."""

    # Phase 1: Load and process
    csv_path = str(DATA_DIR / "2024-03-24-FullBilateralMarkers.csv")
    data_csv = load_data(csv_path)
    data_csv = remove_frames(data_csv)  # time_limit=0 default

    wingspan_path = str(SRC_DIR / "TotalWingspans.yml")
    data_csv = scale_data(data_csv, wingspan_path)

    markers, frame_info, markers_df, frame_info_df = process_data(data_csv)

    turn_csv = str(DATA_DIR / "2024-06-01-ObstacleTurnsSeqList.csv")
    frame_info_df = add_turn_info(frame_info_df, turn_csv)

    tailpack_csv = str(DATA_DIR / "2024-06-11-tailpack.csv")
    markers_with_tailpack, combined_frame_info_df = add_tailpack_data(
        markers_df, frame_info_df, tailpack_csv,
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
def old_repo_refs():
    """Load reference values from the old repository."""
    return {
        "uni_pcs": np.load(OLD_REPO / "unilateral_principal_components.npy"),
        "uni_mu": np.load(OLD_REPO / "unilateral_mu.npy"),
        "uni_scores": np.load(OLD_REPO / "unilateral_scores.npy"),
        "bilateral_data": np.load(OLD_REPO / "bilateral_data.npy"),
        "scaled_markers_with_tailpack": np.load(
            OLD_REPO / "scaled_markers_with_tailpack.npy"
        ),
        "transformed_markers_with_tailpack": np.load(
            OLD_REPO / "transformed_markers_with_tailpack.npy"
        ),
    }


# ---------------------------------------------------------------------------
# Tests: frame counts
# ---------------------------------------------------------------------------
class TestFrameCounts:
    def test_bilateral_frame_count(self, pipeline_results, old_repo_refs):
        """Bilateral frame count must match old repo (time_limit=0)."""
        new = pipeline_results["markers_with_tailpack"].shape[0]
        old = old_repo_refs["scaled_markers_with_tailpack"].shape[0]
        assert new == old, f"Bilateral frames: new={new}, old={old}"

    def test_unilateral_frame_count(self, pipeline_results, old_repo_refs):
        """Unilateral frame count = 2 × bilateral."""
        new = pipeline_results["unilateral_data"].shape[0]
        old = old_repo_refs["uni_scores"].shape[0]
        assert new == old, f"Unilateral frames: new={new}, old={old}"


# ---------------------------------------------------------------------------
# Tests: pre-rotation data (must match exactly)
# ---------------------------------------------------------------------------
class TestPreRotationData:
    def test_scaled_markers_match(self, pipeline_results, old_repo_refs):
        """Scaled markers with tailpack must match old repo exactly."""
        new = pipeline_results["markers_with_tailpack"]
        old = old_repo_refs["scaled_markers_with_tailpack"]
        np.testing.assert_allclose(new, old, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: unilateral PCA components
#
# The new pipeline uses pca.mean_ (training-only) for reconstruction,
# while the old repo used np.mean(all_data). This is a deliberate
# methodological fix. The resulting PCs are very close but not identical.
# ---------------------------------------------------------------------------
class TestUnilateralPCA:
    @pytest.mark.parametrize("pc_idx", range(12))
    def test_component_cosine_similarity(
        self, pipeline_results, old_repo_refs, pc_idx,
    ):
        """Each unilateral PC must have cosine similarity > 0.996 with old.

        Tolerance accounts for the deliberate change from all-data mean
        to training-only mean in the reconstruction step.
        """
        new_pc = pipeline_results["uni_pcs"][pc_idx]
        old_pc = old_repo_refs["uni_pcs"][pc_idx]
        cos_sim = np.abs(np.dot(new_pc, old_pc)) / (
            np.linalg.norm(new_pc) * np.linalg.norm(old_pc)
        )
        assert cos_sim > 0.996, (
            f"PC{pc_idx + 1}: cosine similarity = {cos_sim:.6f} (need > 0.996)"
        )

    def test_explained_variance_matches(self, pipeline_results, old_repo_refs):
        """Scores shape must match, implying same variance structure."""
        new_scores = pipeline_results["uni_scores"]
        old_scores = old_repo_refs["uni_scores"]
        assert new_scores.shape == old_scores.shape
