# Visualization-Tool

Aplikacja webowa stworzona w Streamlit do interaktywnej analizy i wizualizacji zbiorów danych.

## Funkcjonalności

* Wczytywanie danych z plików `.csv` z niestandardowym separatorem. 
* Modyfikacja danych: próbkowanie, redukcja wymiarowości (`t-SNE`, `UMAP`, `TRIMAP`, `PaCMAP`). 
* Obliczanie statystyk: opisowe, korelacje `Pearsona` i `Spearmana`. 
* Generowanie różnorodnych wykresów: histogramy, punktowe, słupkowe, liniowe, pudełkowe i mapy ciepła. 

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