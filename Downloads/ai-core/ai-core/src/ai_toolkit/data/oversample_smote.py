import pandas as pd
from imblearn.over_sampling import SMOTE
from ._valida import precisa_balancear

def oversample_smote(
    df: pd.DataFrame,
    col: str = "genero",
    strategy: float | dict = 0.5,   # fração ou dict
    seed: int = 42,
    validar: bool = True,
) -> pd.DataFrame:
    """
    SMOTE numérico. Precisa que df já contenha features numéricas vetorizadas
    (p.ex. embeddings) – NÃO IDs de tokenizer crus.
    """
    if validar and not precisa_balancear(df[col]):
        return df

    smote = SMOTE(random_state=seed, sampling_strategy=strategy)
    X = df.drop(columns=[col])
    y = df[col]
    X_res, y_res = smote.fit_resample(X, y)
    return X_res.assign(**{col: y_res})