# src/statistics.py

import pandas as pd
from typing import Literal, List, Optional

def calculate_descriptive_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Oblicza statystyki opisowe dla ramki danych."""
    if columns:
        # Wybierz tylko określone kolumny
        selected_df = df[columns]
    else:
        # Użyj wszystkich kolumn numerycznych
        selected_df = df.select_dtypes(include=['number'])
    
    if selected_df.empty:
        raise ValueError("Brak kolumn numerycznych do obliczenia statystyk opisowych.")
    
    return selected_df.describe().T

def calculate_correlation(df: pd.DataFrame, method: Literal['pearson', 'spearman'], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Oblicza macierz korelacji dla podanej metody."""
    if columns:
        # Wybierz tylko określone kolumny numeryczne
        numeric_df = df[columns].select_dtypes(include=['number'])
    else:
        # Użyj wszystkich kolumn numerycznych
        numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        raise ValueError("Brak kolumn numerycznych do obliczenia korelacji.")
    
    return numeric_df.corr(method=method)