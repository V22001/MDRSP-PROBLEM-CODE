import numpy as np
import cvxpy as cp
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import sqrtm
import matplotlib.lines as mlines
from matplotlib.patches import Patch

# =====================================================================
# 1. Robust Parameters Estimation (Calculates Lambda 1 & Lambda 2)
# =====================================================================
def estimate_robust_params(returns, alpha=0.05):
    N = len(returns)     # Sample size
    k = returns.shape[1] # Number of assets
    
    # Mean Uncertainty (Lambda 1) & Covariance Uncertainty (Lambda 2)
    lambda_1 = chi2.ppf(1 - alpha, df=k) / N
    lambda_2 = (N - 1) / (N - k - 1) 
    
    return lambda_1, lambda_2

# =====================================================================
# 2. Get Bounds for Epsilon (Risk Limits for the Frontier)
# =====================================================================
def get_risk_bounds(mu, Sigma):
    n = len(mu)
    w = cp.Variable(n, nonneg=True)
    
    # Minimum Variance (Leftmost point of frontier)
    prob_min = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), [cp.sum(w) == 1])
    prob_min.solve(solver=cp.SCS) 
    min_risk = prob_min.value
    
    # Variance of Max Return Portfolio (Rightmost point of frontier)
    prob_max_ret = cp.Problem(cp.Maximize(mu @ w), [cp.sum(w) == 1])
    prob_max_ret.solve(solver=cp.SCS)
    max_risk = w.value.T @ Sigma @ w.value
    
    return min_risk, max_risk

# =====================================================================
# 3. Strict SDP DRPO Solver (From Paper's Appendix C.4)
# =====================================================================
def solve_drpo_epigraph(mu_hat, Sigma_hat, lambda_1, lambda_2, epsilon, tau=0.001):
    n = len(mu_hat)
    
    # Standard Decision Variables
    x = cp.Variable(n, nonneg=True)
    Q = cp.Variable((n, n), PSD=True)
    q = cp.Variable(n)
    r = cp.Variable()
    P = cp.Variable((n, n), symmetric=True)
    p = cp.Variable(n)
    s = cp.Variable(nonneg=True)
    
    # Auxiliary Epigraph Variables (Appendix C.4)
    y = cp.Variable() # Bounding variable for portfolio risk
    v = cp.Variable() # Epigraph variable for the inverse barrier
    
    # Strict Linear Objective (Eq 33)
    objective = cp.Minimize(lambda_2 * cp.trace(Sigma_hat @ Q) - 
                            cp.trace(np.outer(mu_hat, mu_hat) @ Q) + 
                            cp.trace(Sigma_hat @ P) - 2 * mu_hat @ p + 
                            lambda_1 * s + r + v)
    
    constraints = [
        cp.sum(x) == 1, 
        p == -0.5 * q - Q @ mu_hat
    ]
    
    # Dual Moment LMIs
    lmi_1 = cp.bmat([[P, cp.reshape(p, (n, 1))], 
                     [cp.reshape(p, (1, n)), cp.reshape(s, (1, 1))]])
    constraints.append(lmi_1 >> 0)

    off_diag = cp.reshape(0.5 * q + 0.5 * x, (n, 1))
    lmi_2 = cp.bmat([[Q, off_diag], 
                     [off_diag.T, cp.reshape(r, (1, 1))]])
    constraints.append(lmi_2 >> 0)

    # Risk Bound LMI (Eq 31)
    Sigma_half = sqrtm(Sigma_hat).real
    I = np.eye(n)
    vec_term = Sigma_half @ x
    lmi_risk = cp.bmat([
        [cp.reshape(y, (1, 1)), cp.reshape(vec_term, (1, n))],
        [cp.reshape(vec_term, (n, 1)), I]
    ])
    constraints.append(lmi_risk >> 0)

    # Barrier LMI (Eq 32)
    sqrt_tau = np.sqrt(tau)
    tau_matrix = np.array([[sqrt_tau]])
    lmi_barrier = cp.bmat([
        [cp.reshape(v, (1, 1)), tau_matrix],
        [tau_matrix, cp.reshape(epsilon - y, (1, 1))]
    ])
    constraints.append(lmi_barrier >> 0)

    # Solve via SCS (optimized for LMIs)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, max_iters=5000, verbose=False) 
        return x.value if prob.status in ["optimal", "optimal_inaccurate"] else None
    except:
        return None

# =====================================================================
# 4. Random Portfolio Generator (For 2D Cloud)
# =====================================================================
def generate_portfolio_cloud(mu, sigma, num_samples=2000):
    n = len(mu)
    all_rets, all_risks = [], []
    for _ in range(num_samples):
        w = np.random.dirichlet(np.ones(n), size=1)[0]
        all_rets.append(np.dot(w, mu))
        all_risks.append(np.sqrt(np.dot(w.T, np.dot(sigma, w))))
    return all_risks, all_rets

# =====================================================================
# 5. 3D Plotting Function for Mean Uncertainty
# =====================================================================
def plot_3d_mean_uncertainty(mu_hat, Sigma_hat, lambda_1, tickers):
    resolution = 50
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    Sigma_half = sqrtm(Sigma_hat).real
    transformation_matrix = np.sqrt(lambda_1) * Sigma_half
    ellipsoid_points = transformation_matrix @ sphere_points

    X = (ellipsoid_points[0, :] + mu_hat[0]).reshape(x.shape)
    Y = (ellipsoid_points[1, :] + mu_hat[1]).reshape(y.shape)
    Z = (ellipsoid_points[2, :] + mu_hat[2]).reshape(z.shape)

    fig1 = plt.figure(figsize=(9, 7))
    ax = fig1.add_subplot(111, projection='3d')

    ax.scatter(*mu_hat, color='darkred', s=100, marker='X', zorder=5, label=r'Empirical Mean ($\hat{\mu}$)')
    ax.plot_surface(X, Y, Z, color='cornflowerblue', alpha=0.5, edgecolor='navy', linewidth=0.3)

    ax.set_title(rf'Mean Uncertainty Set ($\lambda_1 = {lambda_1:.4f}$)', fontsize=14, pad=15)
    ax.set_xlabel(f'\nExpected Return: {tickers[0]}', fontsize=10)
    ax.set_ylabel(f'\nExpected Return: {tickers[1]}', fontsize=10)
    ax.set_zlabel(f'\nExpected Return: {tickers[2]}', fontsize=10)

    legend_elements = [
        mlines.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', markersize=10, label=r'Empirical Mean ($\hat{\mu}$)'),
        Patch(facecolor='cornflowerblue', alpha=0.5, edgecolor='navy', label=r'Ambiguity Set ($\leq \lambda_1$)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.view_init(elev=20, azim=45)
    fig1.tight_layout()

# =====================================================================
# 6. Main Execution Block
# =====================================================================
if __name__ == "__main__":
    tickers = ['NVDA', 'AMD', 'META']
    print(f"Fetching Data for {tickers}...")
    
    data = yf.download(tickers, start="2020-01-01", end="2023-12-31", auto_adjust=True)['Close']
    returns = data.pct_change().dropna()
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252

    # Step A: Calculate Dynamic Parameters
    lambda_1, lambda_2 = estimate_robust_params(returns)
    print("\n--- Calculated Parameters ---")
    print(f"Lambda 1 (Mean Uncertainty)       = {lambda_1:.6f}")
    print(f"Lambda 2 (Covariance Uncertainty) = {lambda_2:.6f}")

    # Step B: Generate 3D Plot (Figure 1: Parameter Space)
    print("\nGenerating 3D Mean Uncertainty Ellipsoid...")
    plot_3d_mean_uncertainty(mu, sigma, lambda_1, tickers)

    # Step C: Generate 2D Pareto Front Data
    print("Generating Feasible Objective Space Cloud...")
    cloud_risk, cloud_ret = generate_portfolio_cloud(mu, sigma)
    
    print("Calculating Risk Bounds for Pareto Frontier...")
    risk_min, risk_max = get_risk_bounds(mu, sigma)
    
    epsilon_values = np.linspace(risk_min * 1.0, risk_max * 1.2, 80)
    frontier_risk, frontier_ret = [], []
    
    print("Solving Exact SDP Formulation for DRPO...")
    for eps in epsilon_values:
        w = solve_drpo_epigraph(mu, sigma, lambda_1, lambda_2, epsilon=eps, tau=0.001) 
        if w is not None:
            frontier_risk.append(np.sqrt(w.T @ sigma @ w))
            frontier_ret.append(w.T @ mu)

    # Step D: Generate 2D Plot (Figure 2: Objective Space)
    fig2 = plt.figure(figsize=(10, 7))
    plt.scatter(cloud_risk, cloud_ret, c='royalblue', s=20, alpha=0.5, label='Feasible Portfolios')
    plt.plot(frontier_risk, frontier_ret, color='red', marker='o', markersize=5, linewidth=2, label='Robust Pareto Frontier')
    
    plt.title(f'DRPO: Feasible Space and Pareto Frontier ({", ".join(tickers)})', fontsize=14)
    plt.xlabel('Portfolio Risk ($\sigma$)', fontsize=12)
    plt.ylabel('Expected Annual Return ($E[R]$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    fig2.tight_layout()
    
    print("\n--- Summary Output ---")
    df_summary = pd.DataFrame({'Risk (Sigma)': frontier_risk, 'Return (E[R])': frontier_ret})
    print(df_summary.tail(5))
    
    # Show both figures simultaneously at the very end
    print("\nDone! Opening visualizations...")
    plt.show()