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
    """
    Redukuje wymiarowość danych numerycznych i łączy wynik z oryginalnymi danymi nienumerycznymi.
    """
    
    # --- ZMIANA 1: Rozdzielenie danych na numeryczne i nienumeryczne ---
    non_numeric_df = df.select_dtypes(exclude=['number'])
    numeric_df_raw = df.select_dtypes(include=['number'])
    
    # Usuń NaN tylko z danych numerycznych, które pójdą do modelu
    numeric_df_clean = numeric_df_raw.dropna()
    
    if numeric_df_clean.empty:
        raise ValueError("Brak kolumn numerycznych do redukcji wymiarowości po usunięciu NaN.")
    
    n_components = params.get('n_components', 2)
    n_samples = len(numeric_df_clean)
    n_features = len(numeric_df_clean.columns)
    
    # Walidacja podstawowa
    if n_samples < 4:
        raise ValueError(f"Za mało próbek ({n_samples}) po usunięciu NaN. Potrzebne minimum 4 próbki.")
    if n_features < 2:
        raise ValueError(f"Za mało cech ({n_features}). Potrzebne minimum 2 cechy.")
    
    # Przygotuj parametry dla każdej metody (bez zmian)
    if method == 't-SNE':
        if n_components > 3:
            tsne_method = 'exact'
        else:
            tsne_method = 'barnes_hut'
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, method=tsne_method, random_state=42, init='random')
        
    elif method == 'UMAP':
        n_neighbors = min(15, max(2, n_samples - 1))
        reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42, min_dist=0.1)
        
    elif method == 'TRIMAP':
        if not TRIMAP_AVAILABLE: raise ValueError("TRIMAP nie jest dostępny.")
        n_inliers = min(10, max(1, n_samples // 10))
        n_outliers = min(5, max(1, n_samples // 20))
        reducer = TRIMAP(n_dims=n_components, n_inliers=n_inliers, n_outliers=n_outliers, verbose=False)
        
    elif method == 'PaCMAP':
        if not PACMAP_AVAILABLE: raise ValueError("PaCMAP nie jest dostępny.")
        n_neighbors = min(10, max(2, n_samples - 1))
        reducer = PaCMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        
    else:
        raise ValueError(f"Nieznana metoda redukcji wymiarowości: {method}")
    
    # Normalizacja danych
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df_clean)
    
    # Wykonaj redukcję
    try:
        transformed_data = reducer.fit_transform(scaled_data)
    except Exception as e:
        raise ValueError(f"Błąd podczas redukcji wymiarowości metodą {method}: {str(e)}")
    
    # --- ZMIANA 2: Utworzenie nowej ramki danych i połączenie z danymi nienumerycznymi ---
    # Utwórz ramkę danych z wynikami redukcji, używając indeksu z oczyszczonych danych
    reduced_df = pd.DataFrame(
        transformed_data,
        columns=[f'Komponent_{i+1}' for i in range(n_components)],
        index=numeric_df_clean.index
    )
    
    # Połącz wyniki z oryginalnymi danymi nienumerycznymi
    # join='inner' zapewni, że połączymy tylko te wiersze, które były użyte w redukcji
    final_df = pd.concat([reduced_df, non_numeric_df], axis=1, join='inner')
    
    return final_df

def remove_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Usuwa wybrane kolumny z ramki danych.
    """
    if not columns_to_remove:
        return df
    
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolumny nie istnieją: {', '.join(missing_columns)}")
    
    if len(columns_to_remove) >= len(df.columns):
        raise ValueError("Nie można usunąć wszystkich kolumn")
    
    return df.drop(columns=columns_to_remove)