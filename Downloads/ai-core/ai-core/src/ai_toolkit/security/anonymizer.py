"""Anonimização de dados fiscais sensíveis"""
import hashlib
import re

def anonymize_cnpj(cnpj: str) -> str:
    """Anonimiza CNPJ mantendo formato"""
    if not cnpj:
        return "00.000.000/0000-00"
    hash_val = hashlib.sha256(cnpj.encode()).hexdigest()[:8]
    return f"**.***.***/****-{hash_val[:2]}"

def anonymize_description(desc: str) -> str:
    """Remove possíveis dados sensíveis da descrição"""
    # Remove emails
    desc = re.sub(r'\S+@\S+', '[EMAIL]', desc)
    # Remove CNPJs
    desc = re.sub(r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}', '[CNPJ]', desc)
    # Remove CPFs
    desc = re.sub(r'\d{3}\.\d{3}\.\d{3}-\d{2}', '[CPF]', desc)
    return desc

def anonymize_dataframe(df):
    """Anonimiza DataFrame completo"""
    if 'cnpj' in df.columns:
        df['cnpj'] = df['cnpj'].apply(anonymize_cnpj)
    if 'descricao' in df.columns:
        df['descricao_original'] = df['descricao']  # Backup
        df['descricao'] = df['descricao'].apply(anonymize_description)
    return df
