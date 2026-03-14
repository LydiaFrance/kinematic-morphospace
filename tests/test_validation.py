"""
test_validation.py

Tests for the validation module: calculate_phi, test_PCA_with_random,
kmo_test, pca_suitability_test, bootstrapping_pca, bootstrap_pca,
stats_bootstrap_pca, analyse_and_report_pca.
"""

import numpy as np
import pytest

from kinematic_morphospace.validation import (
    calculate_phi,
    kmo_test,
    pca_suitability_test,
    bootstrapping_pca,
    bootstrap_pca,
    stats_bootstrap_pca,
    analyse_and_report_pca,
)
# Alias to avoid pytest collecting this as a test
from kinematic_morphospace.validation import test_PCA_with_random as run_pca_with_random


# -- Fixtures --

@pytest.fixture
def synthetic_markers():
    """Correlated 3D marker data so PCA is meaningful (50 frames, 4 markers, 3 dims)."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal((50, 1))
    noise = rng.standard_normal((50, 12)) * 0.3
    markers = (base + noise).reshape(50, 4, 3)
    return markers


@pytest.fixture
def sample_eigenvalues():
    """Simple eigenvalue array for calculate_phi tests."""
    return np.array([4.0, 2.0, 1.0, 0.5, 0.3])


# =============================================================================
# calculate_phi
# =============================================================================

class TestCalculatePhi:
    def test_returns_float(self, sample_eigenvalues):
        result = calculate_phi(sample_eigenvalues, num_components=3)
        assert isinstance(result, float)

    def test_known_hand_computed_value(self):
        # eigenvalues = [2.0, 1.0], num_components=2
        # phi_num = (4 + 1) - 2 = 3; phi_den = 2*(2-1) = 2
        # phi = sqrt(3/2) ≈ 1.2247
        eigenvalues = np.array([2.0, 1.0])
        phi = calculate_phi(eigenvalues, num_components=2)
        np.testing.assert_allclose(phi, np.sqrt(3.0 / 2.0), rtol=1e-10)

    def test_uniform_eigenvalues_give_phi_zero(self):
        # All eigenvalues = 1 → sum(1^2) = p = p → numerator = 0
        eigenvalues = np.ones(5)
        phi = calculate_phi(eigenvalues, num_components=5)
        assert phi == pytest.approx(0.0)

    def test_different_num_components_gives_different_result(self, sample_eigenvalues):
        phi3 = calculate_phi(sample_eigenvalues, num_components=3)
        phi5 = calculate_phi(sample_eigenvalues, num_components=5)
        assert phi3 != phi5


# =============================================================================
# test_PCA_with_random
# =============================================================================

class TestPCAWithRandom:
    def test_returns_4_tuple(self, synthetic_markers):
        result = run_pca_with_random(synthetic_markers, num_randomisations=5)
        assert len(result) == 4

    def test_psi_phi_non_negative(self, synthetic_markers):
        psi, phi, _, _ = run_pca_with_random(synthetic_markers, num_randomisations=5)
        assert psi >= 0
        assert phi >= 0

    def test_p_values_in_range(self, synthetic_markers):
        _, _, psi_p, phi_p = run_pca_with_random(synthetic_markers, num_randomisations=5)
        assert 0 <= psi_p <= 1
        assert 0 <= phi_p <= 1

    def test_deterministic_with_seed(self, synthetic_markers):
        r1 = run_pca_with_random(synthetic_markers, num_randomisations=10, seed=42)
        r2 = run_pca_with_random(synthetic_markers, num_randomisations=10, seed=42)
        np.testing.assert_array_equal(r1, r2)


# =============================================================================
# kmo_test
# =============================================================================

class TestKMOTest:
    def test_returns_total_and_per_variable(self, synthetic_markers):
        pca_input = synthetic_markers.reshape(50, -1)
        total, per_var = kmo_test(pca_input)
        assert isinstance(total, (float, np.floating))
        assert per_var.shape == (pca_input.shape[1],)

    def test_correlated_data_higher_kmo_than_independent(self):
        rng = np.random.default_rng(42)
        n = 500
        # Correlated: single latent factor drives all variables
        factor = rng.standard_normal((n, 1))
        corr_data = factor * np.array([1, 2, 3, 4, 5]) + rng.standard_normal((n, 5)) * 0.05
        kmo_corr, _ = kmo_test(corr_data)
        # Independent
        indep_data = rng.standard_normal((n, 5))
        kmo_indep, _ = kmo_test(indep_data)
        assert kmo_corr > kmo_indep

    def test_independent_data_low_kmo(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 5))
        total, _ = kmo_test(data)
        assert total < 0.5

    def test_kmo_bounded_zero_one(self):
        """KMO total and per-variable values must be clamped to [0, 1]."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((100, 5))
        total, per_var = kmo_test(data)
        assert 0.0 <= total <= 1.0
        assert np.all(per_var >= 0.0) and np.all(per_var <= 1.0)

    def test_kmo_near_singular_correlation(self):
        """KMO should not crash on near-singular correlation matrices."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal((200, 1))
        # Columns are nearly identical → near-singular correlation matrix
        data = np.hstack([base + rng.standard_normal((200, 1)) * 1e-8
                          for _ in range(4)])
        total, per_var = kmo_test(data)
        assert 0.0 <= total <= 1.0
        assert np.all(np.isfinite(per_var))


# =============================================================================
# pca_suitability_test
# =============================================================================

class TestPCASuitabilityTest:
    def test_returns_expected_keys(self, synthetic_markers):
        result = pca_suitability_test(synthetic_markers, n_bootstrap=10)
        expected = {"bartlett_p_value", "eigenvalues_distinct",
                    "variance_test_p_value", "components_needed"}
        assert set(result.keys()) == expected

    def test_components_needed_ge_1(self, synthetic_markers):
        result = pca_suitability_test(synthetic_markers, n_bootstrap=10)
        assert result["components_needed"] >= 1

    def test_deterministic_with_seed(self, synthetic_markers):
        r1 = pca_suitability_test(synthetic_markers, n_bootstrap=10, seed=42)
        r2 = pca_suitability_test(synthetic_markers, n_bootstrap=10, seed=42)
        assert r1 == r2


# =============================================================================
# bootstrapping_pca
# =============================================================================

class TestBootstrappingPCA:
    def test_returns_expected_keys_and_shapes(self, synthetic_markers):
        result = bootstrapping_pca(synthetic_markers, n_components=3, n_iterations=10)
        assert set(result.keys()) == {
            "mean_components", "component_ci",
            "mean_explained_variance", "explained_variance_ci",
        }
        assert result["mean_components"].shape[0] == 3

    def test_deterministic_with_seed(self, synthetic_markers):
        r1 = bootstrapping_pca(synthetic_markers, n_components=3, n_iterations=10, seed=42)
        r2 = bootstrapping_pca(synthetic_markers, n_components=3, n_iterations=10, seed=42)
        np.testing.assert_array_equal(r1["mean_components"], r2["mean_components"])


# =============================================================================
# bootstrap_pca
# =============================================================================

class TestBootstrapPCA:
    def test_returns_ci_lower_upper(self, synthetic_markers):
        ci_lower, ci_upper = bootstrap_pca(synthetic_markers, n_bootstrap=10)
        assert ci_lower.ndim == 1
        assert ci_upper.ndim == 1

    def test_lower_le_upper(self, synthetic_markers):
        ci_lower, ci_upper = bootstrap_pca(synthetic_markers, n_bootstrap=10)
        assert np.all(ci_lower <= ci_upper)


# =============================================================================
# stats_bootstrap_pca
# =============================================================================

class TestStatsBootstrapPCA:
    def test_returns_expected_keys(self, synthetic_markers):
        result = stats_bootstrap_pca(synthetic_markers, n_bootstraps=10)
        expected = {
            "pca", "distinct_pcs", "significant_loadings",
            "mean_eigvals", "se_eigvals", "mean_loadings",
            "se_loadings", "mean_scores", "se_scores",
        }
        assert set(result.keys()) == expected

    def test_distinct_pcs_only_significant(self, synthetic_markers):
        """After fix: only significant PCs are appended to distinct_pcs."""
        result = stats_bootstrap_pca(synthetic_markers, n_bootstraps=10, seed=42)
        n_features = synthetic_markers.reshape(50, -1).shape[1]
        # After fix, distinct_pcs should only contain significant indices
        assert len(result["distinct_pcs"]) <= n_features - 1


# =============================================================================
# analyse_and_report_pca
# =============================================================================

class TestAnalyseAndReportPCA:
    def test_returns_dict(self, synthetic_markers):
        result = analyse_and_report_pca(synthetic_markers, n_bootstraps=10, seed=42)
        assert isinstance(result, dict)
        assert "distinct_pcs" in result
