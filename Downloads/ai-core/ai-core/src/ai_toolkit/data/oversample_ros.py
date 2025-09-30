import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from ._valida import precisa_balancear

def oversample_ros(
    df: pd.DataFrame,
    col: str = "genero",
    strategy = 0.5,
    seed: int = 42,
    validar: bool = True,
) -> pd.DataFrame:
    """Oversampling simples via RandomOverSampler."""
    if validar and not precisa_balancear(df[col]):
        return df

    ros = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    X_res, y_res = ros.fit_resample(df, df[col])
    return X_res.assign(**{col: y_res})