"""Single entry point: prints every table of the report."""
import stats
from hedging import changes, out_of_sample

if __name__ == "__main__":
    X = changes()
    oos = out_of_sample(X)
    test = oos.date.unique()
    print(f"Changes {X.index[0]:%Y-%m-%d} to {X.index[-1]:%Y-%m-%d} ({len(X)} days), "
          f"test {test[0]:%Y-%m-%d} to {test[-1]:%Y-%m-%d} ({len(test)} days), "
          f"{oos.hedge.nunique() - 1} hedge sets")
    for title, table in [
            ("PC1 by half-year", stats.volatility(X)),
            ("Residual risk (%), median over hedge sets", stats.residual_risk(oos)),
            ("Against Equal, per-hedge Diebold-Mariano at 5%", stats.dm(oos)),
            ("Calibration, unhedged and two-swap hedges", stats.calibration(oos)),
            ("Shrinkage inflation, medians", stats.inflation(oos))]:
        print(f"\n{title}\n{table.to_string(float_format='{:.3g}'.format)}")
