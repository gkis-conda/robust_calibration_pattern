import numpy as np
from scipy import stats

def compute_kish_weighted_statistics(estimates: np.ndarray, weights: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Computes weighted consensus mean, Kish's effective sample size (N_eff),
    unbiased sample variance, standard error, and Student's t confidence intervals.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of shape (N, P) or (N,) containing parameter estimates across N frames for P parameters.
    weights : np.ndarray
        Array of shape (N,) containing frame weights (e.g., recall R_i).
    alpha : float, optional
        Significance level for confidence interval (default 0.05 for 95% CI).
        
    Returns
    ----------
    dict
        Dictionary containing N_eff, weighted_mean, weighted_std, std_err, and ci_bounds.
    """
    estimates = np.atleast_2d(estimates)  # Shape: (N, P)
    weights = np.asarray(weights, dtype=float).flatten()
    
    N = estimates.shape[0]
    if N < 2:
        raise ValueError("At least 2 valid frames are required for confidence interval estimation.")
        
    # 1. Normalize weights: \tilde{w}_i = w_i / \sum w_k
    w_sum = np.sum(weights)
    if w_sum <= 0:
        raise ValueError("Sum of frame weights must be strictly positive.")
    w_norm = weights / w_sum  # Shape: (N,)
    
    # 2. Kish's effective sample size: N_eff = 1 / \sum \tilde{w}_i^2
    n_eff = 1.0 / np.sum(w_norm ** 2)
    
    # 3. Weighted consensus mean: \hat{\boldsymbol{\theta}} = \sum \tilde{w}_i \hat{\boldsymbol{\theta}}_i
    # Broadcasting w_norm[:, None] across parameters P
    weighted_mean = np.sum(w_norm[:, None] * estimates, axis=0)  # Shape: (P,)
    
    # 4. Unbiased weighted sample variance: 
    # s^2_{eff, j} = (N_eff / (N_eff - 1)) * \sum \tilde{w}_i (\hat{\theta}_{i,j} - \hat{\theta}_j)^2
    dof = n_eff - 1.0
    if dof <= 0:
        raise ValueError(f"Effective degrees of freedom (N_eff - 1 = {dof:.3f}) must be positive.")
        
    residuals_sq = (estimates - weighted_mean) ** 2  # Shape: (N, P)
    weighted_var = (n_eff / dof) * np.sum(w_norm[:, None] * residuals_sq, axis=0)  # Shape: (P,)
    weighted_std = np.sqrt(weighted_var)
    
    # Standard Error of the weighted mean: SE_j = s_{eff, j} / \sqrt{N_eff}
    std_err = weighted_std / np.sqrt(n_eff)
    
    # 5. Student's t critical value and Confidence Intervals: t_{1-\alpha/2, N_eff - 1}
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=dof)
    ci_margin = t_crit * std_err
    
    ci_lower = weighted_mean - ci_margin
    ci_upper = weighted_mean + ci_margin
    
    return {
        "n_raw": N,
        "n_eff": float(n_eff),
        "dof": float(dof),
        "mean": weighted_mean,
        "std": weighted_std,
        "std_err": std_err,
        "ci_margin": ci_margin,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha
    }