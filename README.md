# agent SARSA

Repozytorium zawiera autorską implementację agenta SARSA przeznaczoną do wykorzystania w ramach biblioteki [reinforced-lib](https://reinforced-lib.readthedocs.io/en/latest/index.html).  

## Zawartość repozytorium
W repozytorium znajdują się następujące pliki:
- `sarsa.py` – implementacja agenta SARSA,  
- `main.py` – plik uruchamiający symulację,  
- `ext.py` – plik do podmiany w katalogu `\reinforced-lib\examples\ns-3-ccod`.  

## Wymagania
Do uruchomienia projektu potrzebne są:
- Python **3.9+**  
- [ns-3](https://www.nsnam.org/) (przetestowane na wersji zgodnej z reinforced-lib)  
- [reinforced-lib](https://reinforced-lib.readthedocs.io/en/latest/index.html)  

## Sposób uruchomienia
1. Zainstaluj bibliotekę **reinforced-lib** zgodnie z dokumentacją:  
   [https://reinforced-lib.readthedocs.io/en/latest/index.html](https://reinforced-lib.readthedocs.io/en/latest/index.html)  

2. Uruchom plik `main.py` z odpowiednimi parametrami.  

### Przykładowa komenda:
```bash
python $REINFORCED_LIB/examples/ns-3-ccod/main.py --agent="SARSA" --ns3Path="$YOUR_NS3_PATH"

