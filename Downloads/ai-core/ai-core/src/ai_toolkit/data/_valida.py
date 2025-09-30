import pandas as pd

def precisa_balancear(y: pd.Series, limite: float = 0.30) -> bool:
    """
    True se (menor_freq / maior_freq) < limite.
      • limite=0.30  ⇒ aceita até 30 % da classe majoritária.
    """
    counts = y.value_counts()
    if counts.nunique() == 1:          # só uma classe
        return False
    menor, maior = counts.min(), counts.max()
    return menor / maior < limite