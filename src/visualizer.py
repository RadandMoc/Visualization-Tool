# src/visualizer.py

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

def create_plot(df: pd.DataFrame, plot_type: str, plot_params: dict) -> Figure:
    """Tworzy wykres na podstawie typu i parametrów."""
    
    if plot_type == 'Histogram':
        return px.histogram(df, x=plot_params.get('x_col'), title=f"Histogram dla {plot_params.get('x_col')}")
        
    elif plot_type == 'Wykres punktowy':
        return px.scatter(df, x=plot_params.get('x_col'), y=plot_params.get('y_col'), title=f"Wykres punktowy: {plot_params.get('x_col')} vs {plot_params.get('y_col')}")

    elif plot_type == 'Wykres słupkowy':
        return px.bar(df, x=plot_params.get('x_col'), y=plot_params.get('y_col'), title=f"Wykres słupkowy dla {plot_params.get('y_col')}")

    elif plot_type == 'Wykres liniowy':
        return px.line(df, x=df.index, y=plot_params.get('y_col'), title=f"Wykres liniowy dla {plot_params.get('y_col')}")
        
    elif plot_type == 'Wykres pudełkowy':
        return px.box(df, y=plot_params.get('y_col'), title=f"Wykres pudełkowy dla {plot_params.get('y_col')}")
        
    elif plot_type == 'Mapa ciepła':
        corr_df = plot_params['corr_df']
        return px.imshow(corr_df, text_auto=True, aspect="auto", title="Mapa ciepła korelacji")

    raise ValueError("Nieznany typ wykresu.")