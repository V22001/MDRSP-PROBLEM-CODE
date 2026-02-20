import cvxpy as cp
import numpy as np
import scipy.linalg
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import numpy as np
import cvxpy as cp
import scipy.linalg
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import chi2
import warnings

warnings.filterwarnings("ignore")

# Suppress warnings
warnings.filterwarnings("ignore")
# --- 1. Parameters Estimation (Same as Code 1) ---
def estimate_robust_params(returns, alpha=0.05):
    N = len(returns)
    k = returns.shape[1]
    # Gamma1: Mean uncertainty
    gamma1 = chi2.ppf(1 - alpha, df=k) / N
    # Gamma2: Covariance uncertainty (> 1)
    gamma2 = (N - 1) / (N - k - 1)
    return gamma1, gamma2

def solve_eqn8_robust_final(mu_hat, Sigma_hat, limit_eps, gamma1, gamma2, tau_val):
    mu_hat = np.array(mu_hat, dtype=float).reshape(-1, 1)
    n = mu_hat.shape[0]
    Sigma_hat = np.atleast_2d(np.array(Sigma_hat, dtype=float))
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2
    
    try:
        Theta = np.linalg.pinv(Sigma_hat)
        Sigma_sqrt = scipy.linalg.sqrtm(Sigma_hat).real
    except:
        Sigma_hat += 1e-4 * np.eye(n)
        Theta = np.linalg.pinv(Sigma_hat)
        Sigma_sqrt = scipy.linalg.sqrtm(Sigma_hat).real

    theta_mu = Theta @ mu_hat
    term_scalar_ellipsoid = (mu_hat.T @ Theta @ mu_hat).item() - 1.0

    x = cp.Variable((n, 1))
    Q = cp.Variable((n, n), PSD=True)
    q = cp.Variable((n, 1))
    r = cp.Variable()
    t = cp.Variable()
    lam = cp.Variable(nonneg=True)

    term_quad = cp.quad_form(x, Sigma_hat)
    gap = limit_eps - term_quad
    barrier = cp.inv_pos(gap)
    
    objective = cp.Minimize(r + t + tau_val * barrier)

    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        term_quad <= limit_eps 
    ]

    term_trace = cp.trace((gamma2 * Sigma_hat + mu_hat @ mu_hat.T) @ Q)
    term_linear_t = mu_hat.T @ q
    inner_vec = q + 2 * (Q @ mu_hat)
    term_norm = np.sqrt(gamma1) * cp.norm(Sigma_sqrt @ inner_vec, 2)
    constraints.append(t >= term_trace + term_linear_t + term_norm)

    block11_LHS = Q
    block12_LHS = 0.5 * (q + x)
    block22_LHS = cp.reshape(r, (1, 1))
    
    matrix_LHS = cp.vstack([
        cp.hstack([block11_LHS, block12_LHS]),
        cp.hstack([block12_LHS.T, block22_LHS])
    ])

    block11_Ell = Theta
    block12_Ell = -theta_mu
    block22_Ell = cp.reshape(term_scalar_ellipsoid, (1, 1))
    
    matrix_Ell = cp.vstack([
        cp.hstack([block11_Ell, block12_Ell]),
        cp.hstack([block12_Ell.T, block22_Ell])
    ])

    constraints.append(matrix_LHS + lam * matrix_Ell >> 0)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            act_risk = (x.value.T @ Sigma_hat @ x.value).item()
            rob_val = r.value + t.value
            return -rob_val, act_risk
        else:
            return None, None
    except:
        return None, None

# --- Main Execution ---
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
data = yf.download(tickers, start="2020-01-01", end="2023-12-31", progress=False)['Close']
returns_df = data.pct_change().dropna()

SCALE = 1000
mu0 = returns_df.mean().values * SCALE
Sigma0 = (returns_df.cov().values * (SCALE**2)) + 1e-4 * np.eye(len(tickers))

# Risk Range
w_min = cp.Variable(len(tickers))
prob_min = cp.Problem(cp.Minimize(cp.quad_form(w_min, Sigma0)), [cp.sum(w_min)==1, w_min>=0])
prob_min.solve()
min_risk_val = prob_min.value
max_risk_val = np.max(np.diag(Sigma0))

# --- MODIFICATION: Multiple EPS Values ---
# Hum 4 alag alag EPS values le rahe hain min aur max risk ke beech
eps_values = np.linspace(min_risk_val + 10, max_risk_val - 10, 4)
tau_values = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0]

all_results = {}

for eps in eps_values:
    print(f"\n--- Analysis for EPS: {eps:.2f} ---")
    print(f"{'Tau':<10} | {'Return':<12} | {'Act. Risk':<12} | {'Risk Gap':<12}")
    print("-" * 55)
    # g1, g2 = estimate_robust_params()
    current_eps_results = []
    for t_val in tau_values:
        ret, risk = solve_eqn8_robust_final(mu0, Sigma0, eps, gamma1=0.001, gamma2=1.001, tau_val=t_val)
        
        if ret is not None:
            gap = eps - risk
            current_eps_results.append({'tau': t_val, 'return': ret, 'risk': risk, 'gap': gap})
            print(f"{t_val:<10} | {ret:<12.4f} | {risk:<12.2f} | {gap:<12.4f}")
    
    all_results[eps] = current_eps_results

# --- MODIFICATION: Visualization for each EPS ---
for eps, res_list in all_results.items():
    if res_list:
        res_df = pd.DataFrame(res_list)
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Return
        plt.subplot(1, 2, 1)
        plt.plot(res_df['tau'], res_df['return'], marker='o', label=f'EPS={eps:.1f}')
        plt.xscale('log')
        plt.title(f'Tau vs Return (EPS: {eps:.1f})')
        plt.xlabel('Tau')
        plt.ylabel('Return')
        plt.grid(True)

        # Plot 2: Risk Gap
        plt.subplot(1, 2, 2)
        plt.plot(res_df['tau'], res_df['gap'], marker='s', color='r')
        plt.xscale('log')
        plt.title(f'Tau vs Risk Gap (EPS: {eps:.1f})')
        plt.xlabel('Tau')
        plt.ylabel('Gap (EPS - Risk)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
