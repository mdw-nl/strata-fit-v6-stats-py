import pandas as pd
from vantage6.algorithm.tools.decorators import data

from .logic import compute_partial_stats
from .types import PartialStatsOutput, enforce_output_schema

@data(1)
@enforce_output_schema(PartialStatsOutput)
def partial_stats(df: pd.DataFrame):
    return compute_partial_stats(df)
