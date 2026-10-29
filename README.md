# Covariance Estimators and Out-of-Sample Hedging of SOFR Swap Risk

This study evaluates how covariance estimators affect the out-of-sample hedging and risk forecasting of a SOFR swap portfolio. A DV01 profile over 14 swap tenors is hedged with one or two swaps by principal component neutralization or variance minimization, with the covariance matrix estimated each day by the equal-weighted sample, the Ledoit-Wolf and oracle approximating shrinkage (OAS) estimators and exponential weighting at decay factors of 0.94 and 0.98906. 

## Findings

The hedges are evaluated out of sample over 2025. No estimator improves hedging consistently. Shrinkage leaves principal component hedges unchanged and worsens two-swap minimum-variance hedges on up to 38 of 79 pairs, while exponential weighting improves 36 or 37 of the 79 two-swap principal component hedges and worsens 6 to 10. The estimator matters for the risk forecast. Equal weighting overstates the unhedged risk as volatility falls in 2025, exponential weighting with a decay factor of 0.94 shows no significant bias, and shrinkage overstates the residual risk of hedged portfolios by a factor of 2.5 to 4.

## Layout

```
data/     committed SOFR swap rates
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/swap_rates.csv` holds daily SOFR swap rates at 14 tenors from 2 to 30 years, from 2 January 2024 to 31 December 2025, from Bloomberg. 

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/main.py
```

This prints every table of the report.
