# Visualization-Tool

Aplikacja webowa stworzona w Streamlit do interaktywnej analizy i wizualizacji zbiorów danych.

## Funkcjonalności

* Wczytywanie danych z plików `.csv` z niestandardowym separatorem. [cite: 8]
* Modyfikacja danych: próbkowanie, redukcja wymiarowości (`t-SNE`, `UMAP`, `TRIMAP`, `PaCMAP`). [cite: 2, 4]
* Obliczanie statystyk: opisowe, korelacje `Pearsona` i `Spearmana`. [cite: 6, 7]
* Generowanie różnorodnych wykresów: histogramy, punktowe, słupkowe, liniowe, pudełkowe i mapy ciepła. [cite: 3]

## Instalacja

1.  Sklonuj repozytorium:
    ```bash
    git clone <adres-repozytorium>
    cd wizualizacja-danych
    ```

2.  (Opcjonalnie, ale zalecane) Stwórz i aktywuj wirtualne środowisko:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Na Windows: venv\Scripts\activate
    ```

3.  Zainstaluj wymagane pakiety:
    ```bash
    pip install -r requirements.txt
    ```

## Uruchomienie

Aby uruchomić aplikację, wykonaj polecenie w głównym folderze projektu:

```bash
streamlit run app.py
```

---

### **Podsumowanie i Typowanie**

Powyższa struktura i kod realizują wszystkie założenia z dokumentacji, w tym:
* **Strumień pracy**: Wczytaj -> Wybierz Akcję (Modyfikuj, Oblicz, Wizualizuj) -> Dodaj blok z wynikiem. [cite: 10, 11]
* **Modyfikacje**: Zmiany w danych są permanentne w ramach sesji poprzez nadpisywanie `st.session_state.data`. [cite: 12]
* **Interfejs**: Logika jest oddzielona, co ułatwia testowanie i rozbudowę.
* **Profesjonalizm**: Użyto typowania (np. `pd.DataFrame`, `List`, `Dict`), modularnej struktury, obsługi błędów i czytelnych komentarzy.

To solidna baza, którą można łatwo rozbudować o dodatkowe funkcje, takie jak bardziej zaawansowane opcje parametryzacji algorytmów czy nowe typy wykresów.