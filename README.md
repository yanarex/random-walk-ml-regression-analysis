# Random Walk Simulation & Bike Sharing Regression Analysis

A Jupyter notebook covering two machine learning and statistics topics: a custom 1D random walk simulator with empirical experiments, and from-scratch linear regression on the UCI Bike Sharing Dataset.

## Notebook

Open `Python-Assignment-1.ipynb` in Jupyter or view it directly on GitHub — all graphs are pre-rendered and visible without running any code.

## Contents

### Problem 1 — Random Walk Simulation

Implements a `RandomWalk1D` class that simulates one-dimensional random walks with configurable step probability and random seed.

**Experiments:**
- **Bias effect** — compares sample paths and final-position distributions for p=0.50 vs p=0.55 over 500 steps
- **Spread vs steps** — plots standard deviation of final positions for N in {100, 200, ..., 1000} with box-and-whisker plots
- **First hitting time** — estimates average time to reach boundary +/-a for a in {1, ..., 50}

### Problem 2 — Regression on Real Data

Uses the Bike Sharing Dataset (`hour.csv`) to predict hourly bike rental counts.

**Parts:**
- Data loading, cleaning, and feature engineering with pandas
- From-scratch `LinearRegression` class using the normal equation (NumPy only)
- Model evaluation: MSE, RMSE, R-squared
- Ridge regression with regularization
- Feature selection and model comparison

## Requirements

```bash
pip install jupyter numpy pandas matplotlib scikit-learn python-dateutil
```

## Run the Notebook

```bash
jupyter notebook "Python-Assignment-1.ipynb"
```

All plots are already embedded — you can view them on GitHub without running anything.

## Dataset

`hour.csv` — Hourly bike sharing counts from the Capital Bikeshare system in Washington D.C. (2011-2012).
Source: UCI ML Repository (https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

## Author

Tony Arrington — Spring 2026
