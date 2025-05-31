# src/data_modifier.py

import pandas as pd
from typing import Literal, Dict, Any

# Redukcja wymiarowości
from sklearn.manifold import TSNE
from umap import UMAP
from trimap import TRIMAP
from pacmap import PaCMAP

def sample_data(df: pd.DataFrame, method: Literal['Pierwsze n', 'Ostatnie n', 'Losowe n'], n_samples: int) -> pd.DataFrame:
    """Próbkuje dane zgodnie z wybraną metodą. [cite: 4]"""
    if n_samples >= len(df):
        return df
    if method == 'Pierwsze n':
        return df.head(n_samples)
    if method == 'Ostatnie n':
        return df.tail(n_samples)
    if method == 'Losowe n':
        return df.sample(n=n_samples)
    return df

def reduce_dimensions(df: pd.DataFrame, method: Literal['t-SNE', 'UMAP', 'TRIMAP', 'PaCMAP'], params: Dict[str, Any]) -> pd.DataFrame:
    """Redukuje wymiarowość danych przy użyciu wybranej metody. [cite: 2]"""
    reducers = {
        't-SNE': TSNE,
        'UMAP': UMAP,
        'TRIMAP': TRIMAP,
        'PaCMAP': PaCMAP
    }
    
    n_components = params.pop('n_components', 2)
    reducer = reducers[method](n_components=n_components, **params)
    
    transformed_data = reducer.fit_transform(df.select_dtypes(include=['number']))
    
    return pd.DataFrame(
        transformed_data,
        columns=[f'Komponent_{i+1}' for i in range(n_components)],
        index=df.index
    )