import numpy as np
import pandas as pd

from hypothesis_testing.cointegration.utils_parallel import test_baskets_cointegration_parallel as run_parallel_cointegration


def test_parallel_accepts_optional_regime_args():
    # Minimal synthetic two-symbol dataset with sufficient rows
    idx = pd.date_range("2024-01-01", periods=200, freq="15min")
    a = 100 * np.exp(np.cumsum(np.random.normal(0, 0.005, size=len(idx))))
    b = 50 * np.exp(np.cumsum(np.random.normal(0, 0.005, size=len(idx))))
    df = pd.DataFrame({"AAA_close": a, "BBB_close": b}, index=idx)

    baskets = [["AAA", "BBB"]]

    # Call without regime labels (baseline path)
    res1 = run_parallel_cointegration(df, baskets, max_workers=1, batch_size=1, deduplicate=False)
    assert isinstance(res1, list)

    # Call with regime args as None (should behave the same)
    res2 = run_parallel_cointegration(
        df,
        baskets,
        regime_labels=None,
        include_states=None,
        max_workers=1,
        batch_size=1,
        deduplicate=False,
    )
    assert isinstance(res2, list)


