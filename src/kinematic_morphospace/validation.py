"""Statistical validation and robustness testing for PCA results.

Includes permutation tests, bootstrap confidence intervals,
KMO sampling adequacy, and eigenvalue distinctness checks.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample
import scipy.stats as stats
from tqdm import tqdm


from .pca_core import get_PCA_input



# ------ Testing PCA with Bootstrapping etc ---------

def calculate_phi(eigenvalues, num_components):
    """Compute the Phi statistic for the first *num_components* eigenvalues.

    Phi measures the departure of the leading eigenvalues from a
    uniform (spherical) distribution.  It is defined as::

        Phi = sqrt( (sum(lambda_i^2) - p) / (p * (p - 1)) )

    where *p* = ``num_components`` and the sum runs over the first *p*
    eigenvalues.  The formula originates from correlation-matrix PCA
    (Gleason & Staelin, 1975) where eigenvalues of uncorrelated data
    equal 1 and Phi → 0.

    Note: When applied to covariance-matrix PCA (as in this pipeline)
    the absolute Phi value is not directly interpretable, but the
    permutation comparison (real Phi vs shuffled Phi) remains valid.
    """
    p_num_traits = num_components
    phi_numerator = np.sum(eigenvalues[:num_components]**2) - p_num_traits
    phi_denominator = p_num_traits * (p_num_traits - 1)
    phi = np.sqrt(abs(phi_numerator) / phi_denominator)
    return phi

def test_PCA_with_random(markers, num_randomisations=1000, num_components=5, seed=None):
    """Test PCA significance via column-wise permutation (randomisation test).

    Computes Psi and Phi statistics on the real data and compares them
    against a null distribution obtained by independently shuffling each
    column of the PCA input.

    Parameters
    ----------
    markers : np.ndarray
        Marker array of shape ``(n_frames, n_markers, 3)``.
    num_randomisations : int, optional
        Number of permutation iterations (default 1000).
    num_components : int, optional
        Number of leading components used for the Phi statistic.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    psi : float
        Psi statistic for the real data.
    phi : float
        Phi statistic for the first *num_components* components.
    psi_p_value : float
        Proportion of randomised Psi values >= the real Psi.
    phi_p_value : float
        Proportion of randomised Phi values >= the real Phi.
    """
    rng = np.random.default_rng(seed)
    pca_input = get_PCA_input(markers)

    # Run PCA
    pca = PCA()
    pca_output = pca.fit(pca_input)

    # Calculate the eigenvalues
    eigenvalues = pca_output.explained_variance_

    explained_variance_ratio = pca_output.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    standard_errors = np.sqrt(2 * explained_variance_ratio / pca_input.shape[0])

    print(f"Eigenvalues: {eigenvalues}")

    for i, (var, cum_var, se) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio, standard_errors)):
        print(f"PC{i+1}: Variance explained: {var:.4f}, Cumulative variance explained: {cum_var:.4f}, SE: {se:.4f}")

    # Psi = Σ(λᵢ - 1)²  — designed for correlation-matrix PCA where
    # eigenvalues of uncorrelated data = 1.  With covariance-matrix PCA
    # the "minus 1" has no normalised interpretation, but the
    # permutation comparison (real Psi vs shuffled Psi) is still valid.
    psi = np.sum((eigenvalues - 1)**2)

    # Calculate the phi statistic for the specified number of components
    phi = calculate_phi(eigenvalues, num_components)

    print(f"Phi (first {num_components} PCs): {phi}")

    # Randomisation test for Psi and Phi
    psi_randomised = []
    phi_randomised = []
    for _ in range(num_randomisations):
        randomised_data = np.apply_along_axis(rng.permutation, 0, pca_input)
        randomised_pca_output = PCA().fit(randomised_data)
        randomised_eigenvalues = randomised_pca_output.explained_variance_

        psi_rand = np.sum((randomised_eigenvalues - 1)**2)
        phi_rand = calculate_phi(randomised_eigenvalues, num_components)

        psi_randomised.append(psi_rand)
        phi_randomised.append(phi_rand)

    # Compare actual Psi and Phi to the randomised values
    psi_p_value = np.mean(np.array(psi_randomised) >= psi)
    phi_p_value = np.mean(np.array(phi_randomised) >= phi)

    # Calculate and present significant loadings
    loadings = pca_output.components_.T
    significant_loadings = loadings[np.abs(loadings) > 0.5]
    print(f"Significant Loadings: {significant_loadings}")

    # Calculate and present standard errors on PC-scores
    pc_scores = pca_output.transform(pca_input)
    pc_scores_se = np.std(pc_scores, axis=0) / np.sqrt(len(pc_scores))

    for i, se in enumerate(pc_scores_se):
        print(f"PC{i+1} Scores SE: {se:.4f}")

    return psi, phi, psi_p_value, phi_p_value


def kmo_test(data):
    """Compute the Kaiser-Meyer-Olkin (KMO) sampling adequacy index.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    kmo_total : float
        Overall KMO index.
    kmo_per_variable : np.ndarray
        Per-variable KMO values.
    """
    n = data.shape[1]
    corr = np.corrcoef(data.T)
    corr = np.where(np.eye(n) > 0, 0, corr)
    try:
        partial_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        partial_corr = np.linalg.pinv(corr)
    np.fill_diagonal(partial_corr, 0)
    A = np.sum(corr**2, axis=1) - n
    B = np.sum(partial_corr**2, axis=1) - n
    kmo_per_variable = np.clip(A / (A + B), 0.0, 1.0)
    kmo_total = np.clip(np.sum(A) / (np.sum(A) + np.sum(B)), 0.0, 1.0)
    return kmo_total, kmo_per_variable

def pca_suitability_test(markers, n_bootstrap=1000, variance_threshold=0.8, alpha=0.05, seed=None):
    rng = np.random.default_rng(seed)
    pca_input = get_PCA_input(markers)

    # 1. KMO Test
    kmo_total, kmo_per_variable = kmo_test(pca_input)

    # 2. Bartlett's Test
    _, p_value = stats.bartlett(*[pca_input[:, i] for i in range(pca_input.shape[1])])

    # 3 & 4. Eigenvalue and Variance Tests
    pca = PCA()
    pca.fit(pca_input)

    eigenvalues = pca.explained_variance_

    # Bootstrap for eigenvalue confidence intervals
    bootstrap_eigenvalues = []
    for _ in range(n_bootstrap):
        X_boot = pca_input[rng.choice(pca_input.shape[0], size=pca_input.shape[0], replace=True)]
        pca_boot = PCA()
        pca_boot.fit(X_boot)
        bootstrap_eigenvalues.append(pca_boot.explained_variance_)
    
    bootstrap_eigenvalues = np.array(bootstrap_eigenvalues)
    
    # Check for distinctness
    ci_lower = np.percentile(bootstrap_eigenvalues, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_eigenvalues, 97.5, axis=0)
    
    distinct = np.all(ci_upper[:-1] < ci_lower[1:])
    
    # Variance explained test
    cumulative_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = np.argmax(cumulative_var >= variance_threshold) + 1
    
    var_test_statistic = (cumulative_var[k-1] - variance_threshold) / np.std(np.cumsum(bootstrap_eigenvalues, axis=1)[:, k-1] / np.sum(bootstrap_eigenvalues, axis=1))
    var_p_value = 1 - stats.norm.cdf(var_test_statistic)
    
    return {
        "bartlett_p_value": p_value,
        "eigenvalues_distinct": distinct,
        "variance_test_p_value": var_p_value,
        "components_needed": k
    }



def bootstrapping_pca(markers, n_components, n_iterations=1000, confidence_level=0.95, seed=None):
    rng = np.random.default_rng(seed)
    pca_input = get_PCA_input(markers)

    n_samples, n_features = pca_input.shape

    # Initialize arrays to store results
    all_components = np.zeros((n_iterations, n_components, n_features))
    all_explained_variances = np.zeros((n_iterations, n_components))

    for i in range(n_iterations):
        # Resample with replacement
        indices = rng.integers(0, n_samples, n_samples)
        X_resampled = pca_input[indices]
        
        # Perform PCA on resampled data
        pca = PCA(n_components=n_components)
        pca.fit(X_resampled)
        
        all_components[i] = pca.components_
        all_explained_variances[i] = pca.explained_variance_ratio_
    
    # Calculate confidence intervals
    ci_lower = (1 - confidence_level) / 2
    ci_upper = 1 - ci_lower
    
    component_ci = np.percentile(all_components, [ci_lower * 100, ci_upper * 100], axis=0)
    explained_variance_ci = np.percentile(all_explained_variances, [ci_lower * 100, ci_upper * 100], axis=0)
    
    return {
        'mean_components': np.mean(all_components, axis=0),
        'component_ci': component_ci,
        'mean_explained_variance': np.mean(all_explained_variances, axis=0),
        'explained_variance_ci': explained_variance_ci
    }


def bootstrap_pca(markers, n_bootstrap=1000, ci=95, seed=None):
    rng = np.random.default_rng(seed)
    pca_input = get_PCA_input(markers)

    n_samples, n_features = pca_input.shape
    all_eigenvalues = []

    for _ in range(n_bootstrap):
        # Resample the data with replacement
        indices = rng.integers(0, n_samples, n_samples)
        boot_sample = pca_input[indices]
        
        # Perform PCA on the bootstrap sample
        pca = PCA()
        pca.fit(boot_sample)
        
        # Store the eigenvalues (explained variance ratio)
        all_eigenvalues.append(pca.explained_variance_ratio_)
    
    # Convert to numpy array for easy manipulation
    all_eigenvalues = np.array(all_eigenvalues)
    
    # Calculate confidence intervals
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    
    ci_lower = np.percentile(all_eigenvalues, lower_percentile, axis=0)
    ci_upper = np.percentile(all_eigenvalues, upper_percentile, axis=0)
    
    return ci_lower, ci_upper


def stats_bootstrap_pca(markers, n_bootstraps=1000, alpha=0.05, seed=None):
    rng = np.random.default_rng(seed)
    pca_input = get_PCA_input(markers)

    n_samples, n_features = pca_input.shape
    pca = PCA()
    pca.fit(pca_input)

    # Store bootstrap results
    bootstrap_eigvals = np.zeros((n_bootstraps, n_features))
    bootstrap_loadings = np.zeros((n_bootstraps, n_features, n_features))
    bootstrap_scores = np.zeros((n_bootstraps, n_samples, n_features))

    for i in tqdm(range(n_bootstraps)):
        # Bootstrap resampling
        indices = rng.integers(0, n_samples, n_samples)
        boot_data = pca_input[indices]
        
        # Perform PCA on bootstrap sample
        boot_pca = PCA().fit(boot_data)
        bootstrap_eigvals[i] = boot_pca.explained_variance_
        bootstrap_loadings[i] = boot_pca.components_.T
        bootstrap_scores[i] = boot_pca.transform(pca_input)

    print(f"Bootstrap eigvals: {bootstrap_eigvals.shape}")

    # Calculate mean and standard errors
    mean_eigvals = np.mean(bootstrap_eigvals, axis=0)
    se_eigvals = np.std(bootstrap_eigvals, axis=0)
    mean_loadings = np.mean(bootstrap_loadings, axis=0)
    se_loadings = np.std(bootstrap_loadings, axis=0)
    mean_scores = np.mean(bootstrap_scores, axis=0)
    se_scores = np.std(bootstrap_scores, axis=0)

    # Test distinctness of eigenvalues
    print("Starting eigenvalue distinctness test.")
    distinct_pcs = []
    for i in range(n_features - 1):
        t_stat = (mean_eigvals[i] - mean_eigvals[i+1]) / np.sqrt(se_eigvals[i]**2 + se_eigvals[i+1]**2)
        p_value = 1 - stats.t.cdf(t_stat, n_bootstraps - 1)
        if p_value < alpha:
            distinct_pcs.append(i)

    # Test significance of loadings
    significant_loadings = np.abs(mean_loadings) > 2 * se_loadings

    return {
        'pca': pca,
        'distinct_pcs': distinct_pcs,
        'significant_loadings': significant_loadings,
        'mean_eigvals': mean_eigvals,
        'se_eigvals': se_eigvals,
        'mean_loadings': mean_loadings,
        'se_loadings': se_loadings,
        'mean_scores': mean_scores,
        'se_scores': se_scores
    }

def analyse_and_report_pca(markers, n_bootstraps=1000, seed=None):
    pca_input = get_PCA_input(markers)
    results = stats_bootstrap_pca(markers, n_bootstraps=n_bootstraps, seed=seed)
    
    print("PCA Analysis Results:")
    print("1. Distinct Principal Components:")
    for pc in results['distinct_pcs']:
        print(f"PC{pc+1}: {results['mean_eigvals'][pc]:.4f} ± {results['se_eigvals'][pc]:.4f}")
        print(f"   Variance explained: {results['pca'].explained_variance_ratio_[pc]*100:.2f}%")
    
    print("\n2. Significant Loadings:")
    for pc in results['distinct_pcs']:
        print(f"\nPC{pc+1}:")
        for i, is_significant in enumerate(results['significant_loadings'][:, pc]):
            if is_significant:
                print(f"   Feature {i+1}: {results['mean_loadings'][i, pc]:.4f} ± {results['se_loadings'][i, pc]:.4f}")
    
    print("\n3. PC Scores (first 5 samples):")
    for i in range(min(5, pca_input.shape[0])):
        print(f"Sample {i+1}:")
        for pc in results['distinct_pcs']:
            print(f"   PC{pc+1}: {results['mean_scores'][i, pc]:.4f} ± {results['se_scores'][i, pc]:.4f}")

    return results

