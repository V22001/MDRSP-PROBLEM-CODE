# MDRSP-PROBLEM-CODE
Python implementation of Inverse-Barrier methods for Multiobjective Distributionally Robust Stochastic Programming (MDRSP) applied to Portfolio Optimization (DRPO).


# Multiobjective Distributionally Robust Stochastic Programming (MDRSP)

## 📌 Overview
This repository contains the Python implementation for the paper **"Inverse-barrier function methods for multiobjective distributionally robust stochastic optimization"**. 

The code provides a unified barrier-based framework to solve multiobjective distributionally robust stochastic programming (MDRSP) problems under moment uncertainty. Building on the classical Delage-Ye ambiguity set, the model captures uncertainty in both the mean and covariance of random parameters.It uses an inverse-barrier scalarization technique to transform the multiobjective formulation into tractable single-objective subproblems.

The primary application demonstrated here is **Distributionally Robust Portfolio Optimization (DRPO)**.

## ✨ Key Features
**Moment-Based Ambiguity Set:** Incorporates data-driven parameter estimation to handle distributional uncertainty for both expected returns ($inline$\lambda_1$inline$) and covariance ($inline$\lambda_2$inline$).
**Strict SDP Reformulation:** Transforms the inner maximization of the robust problem into a tractable Semidefinite Program (SDP) using the Schur Complement and Epigraph trick.
**Robust Pareto Frontier Generation:** Computes the efficient frontier balancing worst-case expected returns and deterministic portfolio risk.
* **Advanced Visualizations:** Includes 3D rendering of the Mean Ambiguity Set and 2D plotting of the Feasible Objective Space against the Robust Pareto Frontier.

## 📦 Prerequisites & Installation
The optimization framework relies heavily on `cvxpy` (using the SCS solver for LMIs) and `yfinance` for empirical market data.

Required Python libraries:
* `numpy`
* `pandas`
* `scipy`
* `cvxpy`
* `matplotlib`
* `yfinance`

## 🚀 How to Run
Execute the main script to fetch historical data (default: NVDA, AMD, META), calculate the robust parameters, solve the SDP, and generate the visualizations.

```bash
python main.py
