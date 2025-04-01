import pandas as pd
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

from .logic import compute_partial_stats

@data(1)
def partial_stats(df: pd.DataFrame, **kwargs):
    
    results = compute_partial_stats(df)
    info(results)
    return results
