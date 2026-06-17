

#### Crearea mediului virtual (recomandat)
Deschideți terminal în folderul cu proiectul.
în terminal rulați:
```cmd
python -m venv venv
```
Activare:
```cmd
venv\Scripts\activate
```
#### Instalarea requirements-urilor si modulelor
Instalare:
```cmd
pip install -r requirements.txt
```
#### Rulare
- În fișierul data_builder.py se modifcă RAW_FILE, LABEL_FILE și OUTPUT_CSV cu denumirile fișierelor ce trebuie citite.
- Se execută data_builder.py
  În cmd
 ```cmd
py data_builder.py
```

-Apoi se execută train_models.py
