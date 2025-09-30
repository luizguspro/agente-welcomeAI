import numpy as np
import pandas as pd
from ._valida import precisa_balancear


def oversample_faixas(
    df: pd.DataFrame,
    col: str = "genero",
    limites: list[int] | None = None,
    *,
    metodo: str = "deterministico",   # "deterministico" | "aleatorio"
    seed: int | None = None,
    validar: bool = True,
) -> pd.DataFrame:
    """
    Aumenta classes minoritárias até os limiares desejados.

    - deterministico: replica *todas* as linhas de cada classe
      abaixo de cada limiar; equivale à versão original.
    - aleatorio: sorteia apenas o nº de linhas faltantes para
      cada limiar, com reposição, embaralhando o resultado.

    Parâmetros
    ----------
    df : DataFrame de origem
    col : coluna de rótulo
    limites : lista de limiares (ordem irrelevante)
    metodo : "deterministico" ou "aleatorio"
    seed : inteiro para controle de RNG (usado só no modo aleatorio)
    validar : se True, usa `precisa_balancear` antes de inflar
    """
    metodo = metodo.lower()
    if metodo not in {"deterministico", "aleatorio"}:
        raise ValueError("metodo deve ser 'deterministico' ou 'aleatorio'")

    if validar and not precisa_balancear(df[col]):
        return df

    limites = sorted(limites or [200, 140, 70])          # crescente
    partes = [df]

    if metodo == "deterministico":
        # ----- Versão clássica: duplica conjuntos inteiros -----------------
        cont = df[col].value_counts()
        for lim in limites:
            partes.append(df[df[col].isin(cont[cont < lim].index)])
        # Mantém ordenação previsível
        return (
            pd.concat(partes)
            .sort_values(by="ncm_desc")
            .reset_index(drop=True)
        )

    # ----- Método aleatório ------------------------------------------------
    rng = np.random.default_rng(seed)
    cont_atual = df[col].value_counts().to_dict()

    for lim in limites:
        for classe, n_orig in cont_atual.items():
            faltam = max(lim - n_orig, 0)
            if faltam:
                amostras = (
                    df[df[col] == classe]
                    .sample(
                        n=faltam,
                        replace=True,
                        random_state=rng.integers(1e9),
                    )
                )
                partes.append(amostras)
                cont_atual[classe] += faltam

    return (
        pd.concat(partes)
        .sample(frac=1, random_state=rng.integers(1e9))  # embaralha saída
        .reset_index(drop=True)
    )
