import numpy as np
import pandas as pd

# The October scene is missing for about half of 2020 (and 4-6% of 2018/2024), but
# November and December are almost complete. Blanking October for a random share of
# training rows teaches the models a Nov/Dec fallback: in 5 spatial folds this lifted
# macro F1 on October-missing pixels by 0.03-0.07 at no cost on complete pixels.
OCTOBER_DROPOUT = 0.3

def october_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.endswith('_oct') or '_oct_' in column]

def blank_october(
    x: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    x = x.copy()
    hit = rng.random(len(x)) < OCTOBER_DROPOUT
    x.loc[hit, october_columns(list(x.columns))] = np.nan

    return x
