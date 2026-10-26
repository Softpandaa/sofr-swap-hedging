"""Every parameter of the study, declared once."""
from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "swap_rates.csv"

# Sample ----------------------------------------------------------------------
TENORS = ("2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y",
          "25Y", "30Y")
WINDOW = 252                          # estimation window, trading days

# Risk profile ----------------------------------------------------------------
DV01 = (-5000, -2000, -1000, 3500, -100, 1000, 2300, -800, -600, 2500, 3000, 4000,
        1500, 5000)                   # by tenor, $ per bp

# Estimators ------------------------------------------------------------------
EWMA = (0.94, 0.98906)                # RiskMetrics; 63-day half-life

# Hedges ----------------------------------------------------------------------
MAX_COND = 10.0                       # pair kept if cond(V2' B) < MAX_COND in the first window

# Tests -----------------------------------------------------------------------
CRITICAL_T = 1.96                     # two-sided 5% level
