# app.py

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from src.data_loader import load_data
from src.statistics import calculate_descriptive_stats, calculate_correlation
from src.data_modifier import sample_data, reduce_dimensions
from src.visualizer import create_plot

st.set_page_config(layout="wide", page_title="Narzędzie do wizualizacji danych")

# Inicjalizacja stanu sesji
if 'data' not in st.session_state:
    st.session_state.data: pd.DataFrame | None = None
if 'blocks' not in st.session_state:
    st.session_state.blocks: List[Dict[str, Any]] = []

st.title("Wizualizacja Dużych Zbiorów Danych 🚀")
st.markdown("---")

# --- KROK 1: WCZYTANIE DANYCH ---
st.header("1. Wczytaj dane")

# Sprawdzamy, czy dane nie zostały już wczytane, aby uniknąć resetowania przy każdej interakcji
if st.session_state.data is None:
    uploaded_file = st.file_uploader("Wybierz plik .csv", type="csv")
    sep_input = st.text_input(
        "Określ separator", 
        value="auto", 
        help="Wpisz znak separatora (np. ';' lub ','). Wpisz 'auto', aby program spróbował wykryć go automatycznie."
    )

    if st.button("Wczytaj dane") and uploaded_file is not None:
        df = load_data(uploaded_file, separator=sep_input)
        if df is not None:
            st.session_state.data = df
            st.subheader("Podgląd wczytanych danych:")
            st.dataframe(df.head())
            st.rerun() # Odświeżamy, aby ukryć opcje wczytywania i pokazać panel
else:
    st.success("Dane są załadowane. Możesz rozpocząć analizę z panelu bocznego.")
    st.subheader("Podgląd aktualnych danych:")
    st.dataframe(st.session_state.data.head())


if st.session_state.data is not None:
    df = st.session_state.data
    st.sidebar.success("Dane załadowane. Wybierz akcję.")
    
    # --- PANEL STEROWANIA ---
    st.sidebar.header("Panel sterowania")
    action = st.sidebar.radio("Wybierz opcję", ["Modyfikuj dane", "Oblicz statystyki", "Zwizualizuj dane"])

    # ... (reszta kodu app.py pozostaje bez zmian) ...
    if action == "Modyfikuj dane":
        st.sidebar.subheader("Opcje modyfikacji")
        mod_type = st.sidebar.selectbox("Typ modyfikacji", ["Próbkowanie", "Redukcja wymiarowości", "Usuń kolumny"])

        if mod_type == "Próbkowanie": #
            sample_method = st.sidebar.selectbox("Metoda próbkowania", ['Pierwsze n', 'Ostatnie n', 'Losowe n'])
            n_samples = st.sidebar.slider("Liczba wierszy (n)", 1, len(df), 10)
            if st.sidebar.button("Wykonaj próbkowanie"):
                st.session_state.data = sample_data(df, sample_method, n_samples)
                st.session_state.blocks.append({"type": "message", "content": f"Wykonano próbkowanie. Nowa liczba wierszy: {len(st.session_state.data)}.", "title": "Komunikat o modyfikacji"})
                st.rerun()
        
        elif mod_type == "Redukcja wymiarowości": #
            dim_red_method = st.sidebar.selectbox("Algorytm", ['t-SNE', 'UMAP', 'TRIMAP', 'PaCMAP'])
            n_comp = st.sidebar.slider("Docelowa liczba wymiarów", 2, 10, 2)
            if st.sidebar.button("Redukuj wymiary"):
                try:
                    reduced_df = reduce_dimensions(df, dim_red_method, {'n_components': n_comp})
                    st.session_state.data = reduced_df
                    st.session_state.blocks.append({"type": "message", "content": f"Wykonano redukcję wymiarowości metodą {dim_red_method} do {n_comp} wymiarów.", "title": "Komunikat o modyfikacji"})
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Błąd: {e}")

    elif action == "Oblicz statystyki":
        st.sidebar.subheader("Opcje statystyk")
        stat_type = st.sidebar.selectbox("Typ statystyk", ["Statystyki opisowe", "Korelacja"])
        
        # Wybór kolumn
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        all_columns = df.columns.tolist()
        
        if stat_type == "Statystyki opisowe":
            st.sidebar.subheader("Wybór kolumn")
            use_all_numeric = st.sidebar.checkbox("Użyj wszystkich kolumn numerycznych", value=True)
            
            if not use_all_numeric:
                selected_columns = st.sidebar.multiselect(
                    "Wybierz kolumny do analizy:", 
                    options=all_columns,
                    default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
                )
            else:
                selected_columns = None
                
        elif stat_type == "Korelacja":
            st.sidebar.subheader("Wybór kolumn")
            corr_method = st.sidebar.selectbox("Metoda korelacji", ['pearson', 'spearman'])
            use_all_numeric = st.sidebar.checkbox("Użyj wszystkich kolumn numerycznych", value=True)
            
            if not use_all_numeric:
                selected_columns = st.sidebar.multiselect(
                    "Wybierz kolumny do analizy korelacji:", 
                    options=numeric_columns,
                    default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
                )
            else:
                selected_columns = None
        
        if st.sidebar.button("Oblicz"):
            try:
                if stat_type == "Statystyki opisowe":
                    stats_df = calculate_descriptive_stats(df, selected_columns)
                    column_info = f" (kolumny: {', '.join(selected_columns)})" if selected_columns else " (wszystkie kolumny numeryczne)"
                    st.session_state.blocks.append({
                        "type": "dataframe", 
                        "content": stats_df, 
                        "title": f"Statystyki opisowe{column_info}"
                    })
                    
                elif stat_type == "Korelacja":
                    corr_df = calculate_correlation(df, corr_method, selected_columns)
                    column_info = f" (kolumny: {', '.join(selected_columns)})" if selected_columns else " (wszystkie kolumny numeryczne)"
                    st.session_state.blocks.append({
                        "type": "dataframe", 
                        "content": corr_df, 
                        "title": f"Macierz korelacji ({corr_method}){column_info}"
                    })
                    
            except ValueError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"Wystąpił błąd: {e}")
    elif action == "Zwizualizuj dane": #
        st.sidebar.subheader("Opcje wizualizacji")
        plot_type = st.sidebar.selectbox("Typ wykresu", ['Histogram', 'Wykres punktowy', 'Wykres słupkowy', 'Wykres liniowy', 'Wykres pudełkowy', 'Mapa ciepła'])
        
        params = {}
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        all_columns = df.columns.tolist()

        if not numeric_columns:
            st.sidebar.warning("Brak kolumn numerycznych do wizualizacji.")
        else:
            if plot_type in ['Histogram', 'Wykres pudełkowy']:
                params['x_col'] = st.sidebar.selectbox("Wybierz kolumnę", numeric_columns)
                params['y_col'] = params['x_col']
            elif plot_type in ['Wykres punktowy', 'Wykres słupkowy']:
                params['x_col'] = st.sidebar.selectbox("Wybierz kolumnę dla osi X", all_columns)
                params['y_col'] = st.sidebar.selectbox("Wybierz kolumnę dla osi Y", numeric_columns)
            elif plot_type == 'Wykres liniowy':
                params['y_col'] = st.sidebar.selectbox("Wybierz kolumnę do wykreślenia", numeric_columns)
            
            if st.sidebar.button("Generuj wykres"):
                try:
                    if plot_type == 'Mapa ciepła':
                        corr_df = calculate_correlation(df, 'pearson')
                        params['corr_df'] = corr_df
                    
                    fig = create_plot(df, plot_type, params)
                    st.session_state.blocks.append({"type": "plot", "content": fig, "title": f"Wykres: {plot_type}"})
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Błąd podczas generowania wykresu: {e}")

# --- KROK 3: WYŚWIETLANIE BLOKÓW WYNIKOWYCH ---
st.markdown("---")
st.header("Wyniki analizy")

if not st.session_state.blocks:
    st.info("Brak wyników do wyświetlenia. Wykonaj akcję z panelu bocznego.")
else:
    for i, block in enumerate(reversed(st.session_state.blocks)):
        with st.container():
            st.subheader(f"Wynik {len(st.session_state.blocks) - i}: {block['title']}")
            if block['type'] == 'message':
                st.success(block['content'])
            elif block['type'] == 'dataframe':
                st.dataframe(block['content'])
            elif block['type'] == 'plot':
                st.plotly_chart(block['content'], use_container_width=True)
            st.markdown("---")