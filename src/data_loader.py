# src/data_loader.py

import pandas as pd
import streamlit as st
from io import StringIO
from typing import Optional

def detect_and_convert_numeric(df: pd.DataFrame, decimal: str = '.') -> pd.DataFrame:
    """
    Próbuje wykryć i przekonwertować kolumny tekstowe na numeryczne.
    Uwzględnia różne formaty liczb w zależności od separatora dziesiętnego.
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':  # Kolumny tekstowe
            # Próbujemy przekonwertować na numeryczne
            if decimal == ',':
                # Zamień przecinki na kropki dla konwersji
                converted = pd.to_numeric(
                    df_copy[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
            else:
                converted = pd.to_numeric(df_copy[col], errors='coerce')
            
            # Jeśli większość wartości została przekonwertowana, uznajemy kolumnę za numeryczną
            if converted.notna().sum() / len(converted) > 0.5:  # Co najmniej 50% wartości
                df_copy[col] = converted
    
    return df_copy

def load_data(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, separator: str, decimal: str = '.') -> Optional[pd.DataFrame]:
    """
    Wczytuje dane z przesłanego pliku CSV do ramki danych Pandas.
    Jeśli separator to 'auto', próbuje go automatycznie wykryć.
    Pierwsza linia to nagłówek, a pierwsza kolumna to indeks wierszy.

    Args:
        uploaded_file: Plik przesłany przez użytkownika w Streamlit.
        separator: Znak separatora lub 'auto' do automatycznej detekcji.
        decimal: Znak separatora dziesiętnego ('.' lub ',').

    Returns:
        Ramka danych Pandas lub None w przypadku błędu.
    """
    if uploaded_file is None:
        return None

    try:
        string_data = uploaded_file.getvalue().decode('utf-8')
        uploaded_file.seek(0)

        # Jeśli separator to 'auto' lub jest pusty, pandas sam go wykryje ustawiając sep=None
        # Wymaga to użycia silnika 'python'
        effective_sep = None if separator.lower() == 'auto' or not separator else separator
        
        df = pd.read_csv(
            StringIO(string_data),
            sep=effective_sep,
            header=0,
            index_col=0,
            decimal=decimal,  # Używamy wybranego separatora dziesiętnego
            engine='python' # Silnik 'python' jest lepszy w automatycznym wykrywaniu separatora
        )
        
        # Użyj ulepszonej funkcji konwersji
        df = detect_and_convert_numeric(df, decimal)
        
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        total_cols = len(df.columns)
        
        st.success(f"Dane zostały pomyślnie wczytane! (separator dziesiętny: '{decimal}')")
        st.info(f"Wykryto {numeric_cols} kolumn numerycznych z {total_cols} łącznie.")
        
        return df
    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania pliku: {e}")
        st.info("Upewnij się, że pierwsza kolumna może służyć jako unikalny indeks. Jeśli problem nadal występuje, spróbuj ręcznie określić separator i separator dziesiętny.")
        return None