# src/data_loader.py

import pandas as pd
import streamlit as st
from io import StringIO
from typing import Optional

def load_data(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, separator: str) -> Optional[pd.DataFrame]:
    """
    Wczytuje dane z przesłanego pliku CSV do ramki danych Pandas.
    Jeśli separator to 'auto', próbuje go automatycznie wykryć.
    Pierwsza linia to nagłówek, a pierwsza kolumna to indeks wierszy.

    Args:
        uploaded_file: Plik przesłany przez użytkownika w Streamlit.
        separator: Znak separatora lub 'auto' do automatycznej detekcji.

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
            decimal='.',
            engine='python' # Silnik 'python' jest lepszy w automatycznym wykrywaniu separatora
        )
        st.success("Dane zostały pomyślnie wczytane!")
        return df
    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania pliku: {e}")
        st.info("Upewnij się, że pierwsza kolumna może służyć jako unikalny indeks. Jeśli problem nadal występuje, spróbuj ręcznie określić separator (np. ';', ',').")
        return None