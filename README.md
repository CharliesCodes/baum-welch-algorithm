# Baum-Welch-Algorithmus zur Sequenzanalyse in der Bioinformatik

Hierbei handelt es sich um eine Python Implementierung des Baum Welch Algorithmus für Hidden Markov Modelle.<br>
Eine gesuchte Sequenz-Abfolge wird in das Modell gegeben.

Via Baum-Welch Algorithmus werden so lange Forward- & Backward- Algorithmus durchgeführt, bis stabile Resultate entstehen.<br>
Bei jeder Itteration werden mit Hilfe der neu entstandenen Alpha- und Beta-Matrizen die Emissions- und Transmissions Matrizen angepasst.
Jede Baum-Welch Itteration erzeugt dabei fortlaufend nummerierte SVG-Datein, welche jeweils Alpha und Beta Matrix darstellen.

# Eingabe
### Anfangsverteilung/ Initial Probability Distribution π
Die Initialverteilung wurde durch ein zusätzliches Symbol (Komma) im Eingabealphabet hinzugefügt.

### Eingabealphabet  V
Die möglichen Beobachtungen
Festgelegt wurden diese auf die Nukleinsäuren: Adenin, Thymins, Guanin und Cytosin
```python
self.SYMBOLS = [",", "A", "T", "G", "C"]
```

Für die Untersuchung von RNA- oder Protein Sequenzen müssen diese entsprechend angepasst werden
Das Komma dient der Init


### Zustände/ States  S
Das Modell nutzt 3 Zustände  ( 0, 1, 2 )

### Übergangsmatrix/ Transition  A
Die A-Matrix zeigt die Übergangswahrscheinlichkeiten einzelner Zustände ineinander.
Leserichtung: Von Index -> Nach Spalte

|       | **0** | **1** | **2** |
|:-----:|:-----:|:-----:|:-----:|
| **0** |   0   |  0.5  |  0.5  |
| **1** |   0   |  0.25 |  0.75 |
| **2** |  0.5  |  0.25 |  0.25 |


### Beobachtungsmatrix/ Emission  B
Die B-Matrix gibt an, wie wahrscheinlich es ist, die jeweils mögliche Beobachtung in den verschiedenen Zuständen zu erzeugen.<br>Die Komma-Spalte gibt die Anfangsverteilung an.

|       | **,** | **A** | **T** | **G** | **C** |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **0** |   1   |   0   |   0   |   0   |   0   |
| **1** |   0   |  1/3  |  1/3  |  1/6  |  1/6  |
| **2** |   0   |  1/6  |  1/3  |  1/6  |  1/3  |

# Results
Die jeweils hellblau dargestellten Zellen geben die errechnete Wahrscheinlichkeit der gesuchten Output-Sequenz an.

## Forward
![Result Forward](https://github.com/CharliesCodes/baum-welch-algorithm/blob/main/Images/forward.gif)


## Backward
![Result backward](https://github.com/CharliesCodes/baum-welch-algorithm/blob/main/Images/backward.gif)
