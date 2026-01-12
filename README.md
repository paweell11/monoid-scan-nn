# Implementacja i testy modeli neuronowych bazujących na funkcjach monoidalnych

Celem pracy była implementacja modelu uczenia maszynowego opartego na funkcjach monoidalnych oraz porównanie jego efektywności z innymi popularnymi rozwiązaniami. Projekt miał za zadanie weryfikację hipotezy, czy zastosowanie funkcji monoidalnych pozwala połączyć zalety dwóch dominujących architektur w zadaniach przetwarzania sekwencji: równoległość podczas etapu treningu architektury Transformer oraz efektywność inferencji tradycyjnych sieci rekurencyjnych (RNN).

W projekcie za pomocą języka Python zaimplementowano trzy modele uczenia maszynowego: RNN, Transformer oraz model oparty na funkcjach monoidalnych – LRU (Linear Recurrent Unit). Jako problem badawczy wybrano modelowanie języka na poziomie znaków, wykorzystując zbiór danych oparty na powieści „Krzyżacy” Henryka Sienkiewicza. Eksperymenty podzielono na dwie kategorie: analizę jakościową oraz analizę wydajnościową modeli.

W ramach badań jakościowych porównano skuteczność architektur w zależności od liczby parametrów oraz zbadano szybkość zbieżności procesu uczenia, przyjmując za metrykę wartość funkcji straty (entropię krzyżową). Badania wydajnościowe obejmowały natomiast pomiar przepustowości przetwarzania tokenów zarówno podczas treningu, jak i w procesie inferencji.

## Ostateczne wyniki

Wyniki eksperymentów doprowadziły do następujących wniosków:

* **Trening:** Podejście oparte na funkcjach monoidalnych (LRU) skutecznie eliminuje ograniczenia rekurencji, umożliwiając trening z wysoką przepustowością (równoległość).
* **Inferencja:** Model LRU zachowuje stały czas generowania kolejnych tokenów, typowy dla lekkich sieci RNN.
* **Jakość vs Parametry:** Analiza wykazała kompromis architektury LRU – model ten ustępuje konkurencyjnym rozwiązaniom pod względem efektywności parametrowej. Aby uzyskać porównywalne wyniki jakościowe, LRU potrzebuje kilkukrotnie większej liczby parametrów sieci.

Praca pokazuje, że architektury oparte na funkcjach monoidalnych stanowią obiecującą alternatywę w systemach wymagających przetwarzania długich kontekstów.

## Wizualizacja wyników

<img width="1200" height="700" alt="wykres1_4" src="https://github.com/user-attachments/assets/45913b76-ef04-40c0-937f-ff9f3debcea4" />

<img width="1000" height="600" alt="wykres2" src="https://github.com/user-attachments/assets/90d3974b-7dfd-410f-ad10-b98bd1aea4f9" />

<img width="1100" height="700" alt="wykres3" src="https://github.com/user-attachments/assets/3a90cee4-ca62-473e-a134-6958cb503c6e" />

<img width="1000" height="600" alt="wykres4_2" src="https://github.com/user-attachments/assets/33d46234-e920-4f74-81ee-f4dbbc6f4819" />
