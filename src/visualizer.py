# src/visualizer.py

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from typing import Dict

# Mapowanie nazw wykresów na funkcje z plotly.express
PLOT_MAPPING = {
    'Histogram': px.histogram,
    'Wykres punktowy': px.scatter,
    'Wykres słupkowy': px.bar,
    'Wykres liniowy': px.line,
    'Wykres pudełkowy': px.box,
    'Mapa ciepła': px.imshow
}

def create_plot(df: pd.DataFrame, plot_type: str, plot_params: Dict) -> Figure:
    """
    Tworzy wykres dynamicznie na podstawie przekazanych parametrów.
    """
    plot_function = PLOT_MAPPING.get(plot_type)
    if not plot_function:
        raise ValueError(f"Nieznany typ wykresu: '{plot_type}'")
        
    # Obsługa specjalnego przypadku dla mapy ciepła
    if plot_type == 'Mapa ciepła':
        corr_df = plot_params.pop('corr_df', None)
        if corr_df is None:
            raise ValueError("Dla mapy ciepła wymagany jest parametr 'corr_df'.")
        return px.imshow(corr_df, text_auto=True, aspect="auto", **plot_params)

    # Dynamiczne tworzenie wykresu z przekazanych parametrów (np. x, y, color)
    fig = plot_function(df, **plot_params)
    return fig