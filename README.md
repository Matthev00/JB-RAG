# ZPRP-METEO-MODEL


---

## **Opis**



---

## **Kluczowe funkcjonalności**


---

## **Instalacja**


---

## **Sposób użycia**


---

## Organizacja projektu

```
├── LICENSE            <- Licencja Open-Source
├── Makefile           <- Makefile z komendami
├── README.md          <- README 
├── data
│   ├── normalized     <- Znormalizowane dane
│   ├── processed      <- Pobrane i przetworzone dane
│   |── raw            <- Oryginalne dane
|   └── stats.json     <- Dane do normalizacji
│
├── docs               <- Folder z dokumentacją
│   ├── Design_Proposal.md
│   └── Analiza_Literatury.md
│
├── notebooks          <- Notatniki Jupyter. Konwencja nazywania: numer + krótki opis
│   └── 01_exploratory_data_analysis.ipynb
│
├── reports            <- Wygenerowane analizy, np. HTML, PDF
│   |── figures        <- Grafiki i wykresy odnoszące sie do danych i modeli
│   └── training_results <- Analizy eksperymentów 
│
├── requirements.txt   <- Wymagania środowiska
│
├── setup.py           <- Skrypt instalacyjny projektu
├── pyproject.toml     <- Plik konfiguracyjny projektu
│
├── api                <- Kod źródłowy API serwującego predykcję
├── app                <- Kod źródłowy aplikacji wizualizującej predykcję
├── mlruns             <- Folder na wyniki ekspermentów oraz wytrenowane modele 
└── zprp_meteo_model   <- Kod źródłowy projektu(Część "badawcza")
    ├── data/          <- Moduły przetwarzające dane
    ├── model/         <- Kod definiujący architektury modeli
    ├── training/      <- Moduły związane z trenowaniem modeli
    └── utils/         <- Narzędzia wspomagające projekt
```

---

## **Autorzy**

- Mateusz Ostaszewski

---

## **Licencja**



---
