import numpy as np
import cvxpy as cp
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import chi2
from scipy.linalg import sqrtm

# --- 1. Data Setup (Diverse Assets) ---
def get_market_data():
    tickers = ['NVDA', 'TSLA', 'AMD',  
               'PG', 'KO', 'JPM',      
               'GLD', 'SLV', 'SBIN.NS']    
    
    print(f"Downloading Data for {len(tickers)} diverse assets...")
    data = yf.download(tickers, start="2020-01-01", end="2025-01-01", auto_adjust=True)['Close']
    returns = data.pct_change().dropna()
    return returns

# --- 2. Calculate Belief Multipliers (a_i vectors) ---
def get_belief_multipliers(returns):
    mu_long = returns.mean().values * 252
    sigma = returns.std().values * np.sqrt(252)
    mu_short = returns.tail(63).mean().values * 252 
    
    abs_mu = np.abs(mu_long)
    abs_mu[abs_mu < 1e-4] = 1e-4

    vol_ratio = sigma / abs_mu
    a1 = np.maximum(0.5, 1.0 - (0.3 * vol_ratio)) 

    momentum_score = mu_short / abs_mu
    a2 = 1.0 + (momentum_score - 1.0) 
    a2 = np.clip(a2, 0.5, 2.0)

    a3 = 1.0 + (0.5 * vol_ratio)
    
    assets = returns.columns
    print("\n" + "="*60)
    print(f"{'Asset':<10} | {'a1 (Cons)':<12} | {'a2 (Trend)':<12} | {'a3 (Spec)':<12}")
    print("-" * 60)
    for i in range(len(assets)):
         print(f"{assets[i]:<10} | {a1[i]:.4f}       | {a2[i]:.4f}       | {a3[i]:.4f}")
    print("="*60 + "\n")

    return a1, a2, a3, mu_long, returns.cov().values * 252

# --- 3. Robust Parameters ---
def estimate_robust_params(returns, alpha=0.05):
    N = len(returns)
    k = returns.shape[1]
    gamma1 = chi2.ppf(1 - alpha, df=k) / N
    gamma2 = (N - 1) / (N - k - 1)
    
    print(f"Robust Parameters: Gamma1={gamma1:.6f}, Gamma2={gamma2:.6f}")
    return gamma1, gamma2

# --- 4. STRICT LMI SOLVER (Epigraph Formulation for Barrier) ---
def solve_robust_belief_lmi(w_weights, a1, a2, a3, mu_base, Sigma_hat, gamma1, gamma2, epsilon, tau=0.005):
    n = len(mu_base)
    
    # 1. Main Decision Variable
    x = cp.Variable(n, nonneg=True)
    
    # 2. Expert Weighting logic
    w1, w2, w3 = w_weights
    a_combined = (w1 * a1) + (w2 * a2) + (w3 * a3)
    mu_center_val = a_combined * mu_base

    # 3. Robust Dual Variables (Delage-Ye)
    Q = cp.Variable((n, n), PSD=True)
    q = cp.Variable(n)
    r = cp.Variable()
    P = cp.Variable((n, n), symmetric=True)
    p = cp.Variable(n)
    s = cp.Variable(nonneg=True)

    # 4. Epigraph Variables (For Linearizing Barrier Term)
    y = cp.Variable() # Bounds portfolio risk
    v = cp.Variable() # Bounds the barrier penalty

    # Linear Objective (Eq 33 from your paper)
    # Objective = minimize expected loss + barrier penalty
    objective = cp.Minimize(gamma2 * cp.trace(Sigma_hat @ Q) - 
                            cp.trace(np.outer(mu_center_val, mu_center_val) @ Q) + 
                            cp.trace(Sigma_hat @ P) - 2 * mu_center_val @ p + 
                            gamma1 * s + r + v)
    
    constraints = [
        cp.sum(x) == 1, 
        p == -0.5 * q - Q @ mu_center_val
    ]
    
    # LMI 1: Moment constraints
    lmi_1 = cp.bmat([[P, cp.reshape(p, (n, 1))], 
                     [cp.reshape(p, (1, n)), cp.reshape(s, (1, 1))]])
    constraints.append(lmi_1 >> 0)

    # LMI 2: Moment constraints
    off_diag = cp.reshape(0.5 * q + 0.5 * x, (n, 1))
    lmi_2 = cp.bmat([[Q, off_diag], 
                     [off_diag.T, cp.reshape(r, (1, 1))]])
    constraints.append(lmi_2 >> 0)

    # LMI 3: Risk Bound LMI (Eq 31) -> Linearizes Risk
    Sigma_half = sqrtm(Sigma_hat).real
    I = np.eye(n)
    vec_term = Sigma_half @ x
    lmi_risk = cp.bmat([
        [cp.reshape(y, (1, 1)), cp.reshape(vec_term, (1, n))],
        [cp.reshape(vec_term, (n, 1)), I]
    ])
    constraints.append(lmi_risk >> 0)

    # LMI 4: Barrier LMI (Eq 32) -> Linearizes Inverse Barrier
    sqrt_tau = np.sqrt(tau)
    tau_matrix = np.array([[sqrt_tau]])
    lmi_barrier = cp.bmat([
        [cp.reshape(v, (1, 1)), tau_matrix],
        [tau_matrix, cp.reshape(epsilon - y, (1, 1))]
    ])
    constraints.append(lmi_barrier >> 0)

    # Solve via SCS
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, max_iters=5000, verbose=False) 
        if x.value is None:
             return None, None
        return x.value, x.value.T @ mu_center_val
    except:
        return None, None

# --- Main Execution ---
if __name__ == "__main__":
    returns_data = get_market_data()
    a1, a2, a3, mu_base, Sigma = get_belief_multipliers(returns_data)
    g1, g2 = estimate_robust_params(returns_data)
    
    scenarios = [
        ("1. Pure Conservative (a1)", [1.0, 0.0, 0.0]),
        ("2. Pure Trend (a2)",        [0.0, 1.0, 0.0]),
        ("3. Pure Speculative (a3)",  [0.0, 0.0, 1.0]),
        ("4. Balanced",               [0.33, 0.33, 0.34]),
        ("5. Cons/Trend Mix",         [0.5, 0.5, 0.0]),
        ("6. Trend/Spec Mix",         [0.0, 0.5, 0.5]),
        ("7. Cons/Spec Mix",          [0.5, 0.0, 0.5]),
        ("8. Noise A",                [0.1, 0.2, 0.7]),
        ("9. Noise B",                [0.7, 0.2, 0.1]),
        ("10. Noise C",               [0.2, 0.7, 0.1]),
        ("11. Mod Conservative",      [0.6, 0.2, 0.2]),
        ("12. Mod Trend",             [0.2, 0.6, 0.2]),
        ("13. Mod Speculative",       [0.2, 0.2, 0.6]),
        ("14. Extreme Safety",        [0.9, 0.05, 0.05]),
        ("15. Extreme Momentum",      [0.05, 0.9, 0.05]),
        ("16. Extreme Risk",          [0.05, 0.05, 0.9])
    ]
    
    w_dummy = cp.Variable(len(returns_data.columns), nonneg=True)
    cp.Problem(cp.Minimize(cp.quad_form(w_dummy, Sigma)), [cp.sum(w_dummy)==1]).solve(solver=cp.SCS)
    min_risk = np.sqrt(w_dummy.value.T @ Sigma @ w_dummy.value)
    max_risk = np.sqrt(np.diag(Sigma).max()) * 0.8
    
    common_risks = np.linspace(min_risk * 1.05, max_risk, 35)
    epsilon_values = common_risks ** 2
    
    plt.figure(figsize=(14, 9))
    colors = cm.jet(np.linspace(0, 1, len(scenarios)))
    
    print(f"{'Scenario':<25} | Solving with Belief Multipliers & Strict LMI Barrier...")
    
    for i, (label, w) in enumerate(scenarios):
        frontier_risks = []
        frontier_rets = []
        
        for eps in epsilon_values:
            # CALLING THE NEW STRICT LMI SOLVER HERE
            w_opt, ret_val = solve_robust_belief_lmi(w, a1, a2, a3, mu_base, Sigma, g1, g2, eps)
            
            if w_opt is not None:
                act_risk = np.sqrt(w_opt.T @ Sigma @ w_opt)
                frontier_risks.append(act_risk)
                frontier_rets.append(ret_val)
        
        if len(frontier_risks) > 5:
            sorted_indices = np.argsort(frontier_risks)
            plt.plot(np.array(frontier_risks)[sorted_indices], 
                     np.array(frontier_rets)[sorted_indices], 
                     color=colors[i], linewidth=2, label=label, marker='o', markersize=3)

    plt.title(r"Strict LMI MDRSP: Intersecting Frontiers ($h_{ii} = (a_i \odot \xi)^T x$)", fontsize=15)
    plt.xlabel("Portfolio Risk (Standard Deviation $\sigma$)", fontsize=12)
    plt.ylabel("Consensus Expected Return", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()