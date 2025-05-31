# src/data_modifier.py

import pandas as pd
from typing import Literal, Dict, Any, List
import numpy as np

# Redukcja wymiarowości
from sklearn.manifold import TSNE
from umap import UMAP
try:
    from trimap import TRIMAP
    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False
    
try:
    from pacmap import PaCMAP
    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False

def sample_data(df: pd.DataFrame, method: Literal['Pierwsze n', 'Ostatnie n', 'Losowe n'], n_samples: int) -> pd.DataFrame:
    """Próbkuje dane zgodnie z wybraną metodą."""
    if n_samples >= len(df):
        return df
    if method == 'Pierwsze n':
        return df.head(n_samples)
    if method == 'Ostatnie n':
        return df.tail(n_samples)
    if method == 'Losowe n':
        return df.sample(n=n_samples, random_state=42)
    return df

def reduce_dimensions(df: pd.DataFrame, method: Literal['t-SNE', 'UMAP', 'TRIMAP', 'PaCMAP'], params: Dict[str, Any]) -> pd.DataFrame:
    """Redukuje wymiarowość danych przy użyciu wybranej metody."""
    
    # Filtruj tylko kolumny numeryczne i usuń NaN
    numeric_df = df.select_dtypes(include=['number']).dropna()
    if numeric_df.empty:
        raise ValueError("Brak kolumn numerycznych do redukcji wymiarowości.")
    
    n_components = params.get('n_components', 2)
    n_samples = len(numeric_df)
    n_features = len(numeric_df.columns)
    
    # Walidacja podstawowa
    if n_samples < 4:
        raise ValueError(f"Za mało próbek ({n_samples}). Potrzebne minimum 4 próbki.")
    if n_features < 2:
        raise ValueError(f"Za mało cech ({n_features}). Potrzebne minimum 2 cechy.")
    
    # Przygotuj parametry dla każdej metody
    if method == 't-SNE':
        # t-SNE ma ograniczenia dla barnes_hut
        if n_components > 3:
            # Użyj exact dla więcej niż 3 wymiarów
            tsne_method = 'exact'
        else:
            tsne_method = 'barnes_hut'
        
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        reducer = TSNE(
            n_components=n_components, 
            perplexity=perplexity, 
            method=tsne_method,
            random_state=42,
            init='random'
        )
        
    elif method == 'UMAP':
        n_neighbors = min(15, max(2, n_samples - 1))
        reducer = UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors, 
            random_state=42,
            min_dist=0.1
        )
        
    elif method == 'TRIMAP':
        if not TRIMAP_AVAILABLE:
            raise ValueError("TRIMAP nie jest dostępny. Zainstaluj bibliotekę trimap.")
        
        # TRIMAP wymaga konkretnych parametrów
        n_inliers = min(10, max(1, n_samples // 10))
        n_outliers = min(5, max(1, n_samples // 20))
        n_random = min(5, max(1, n_samples // 20))
        
        reducer = TRIMAP(
            n_dims=n_components,
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            n_random=n_random,
            verbose=False
        )
        
    elif method == 'PaCMAP':
        if not PACMAP_AVAILABLE:
            raise ValueError("PaCMAP nie jest dostępny. Zainstaluj bibliotekę pacmap.")
        
        n_neighbors = min(10, max(2, n_samples - 1))
        reducer = PaCMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors,
            random_state=42
        )
        
    else:
        raise ValueError(f"Nieznana metoda redukcji wymiarowości: {method}")
    
    # Normalizacja danych (ważne dla algorytmów redukcji wymiarowości)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Wykonaj redukcję wymiarowości
    try:
        transformed_data = reducer.fit_transform(scaled_data)
    except Exception as e:
        raise ValueError(f"Błąd podczas redukcji wymiarowości metodą {method}: {str(e)}")
    
    return pd.DataFrame(
        transformed_data,
        columns=[f'Komponent_{i+1}' for i in range(n_components)],
        index=numeric_df.index
    )

def remove_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Usuwa wybrane kolumny z ramki danych.
    
    Args:
        df: Ramka danych Pandas
        columns_to_remove: Lista nazw kolumn do usunięcia
    
    Returns:
        Ramka danych bez wybranych kolumn
        
    Raises:
        ValueError: Jeśli próbuje się usunąć wszystkie kolumny lub nieistniejące kolumny
    """
    if not columns_to_remove:
        return df
    
    # Sprawdź czy wszystkie kolumny istnieją
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolumny nie istnieją: {', '.join(missing_columns)}")
    
    # Sprawdź czy nie usuwamy wszystkich kolumn
    if len(columns_to_remove) >= len(df.columns):
        raise ValueError("Nie można usunąć wszystkich kolumn")
    
    return df.drop(columns=columns_to_remove)