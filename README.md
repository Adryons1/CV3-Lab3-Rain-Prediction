# Computer Vision 3 - Laboratorul 3

Tema: predicția ploii din date meteorologice cu PyTorch.

## Fișiere
- `CV3_Lab3_complet.ipynb` - notebook-ul de laborator, cu secțiunea 13 completată și rezultate incluse.
- `lab3_complete.py` - aceeași suită de experimente, rulabilă și ca script.
- `lab-03-NUME-Prenume.pdf` - raportul de laborator (completează numele înainte de predare).

## Rulare rapidă în Google Colab
1. Deschide https://colab.research.google.com/ și folosește **File > Upload notebook** pentru `CV3_Lab3_complet.ipynb`.
2. În Colab, opțional: **Runtime > Change runtime type > T4 GPU**, dacă este disponibil.
3. Notebook-ul profesorului are deja rezultate salvate. Pentru exercițiile completate, mergi la **secțiunea 13** și rulează **de la celula 13.1 în jos**, în ordine. Nu este necesară rerularea secțiunii 8 din exemplul inițial.
4. Datele `weatherAUS.csv` se descarcă automat prin `gdown` dacă nu există în sesiunea Colab.
5. La final, se generează folderul `lab3_results` cu tabelele CSV și `results.json`. Experimentul hardware măsoară CPU și, dacă există CUDA, GPU.

## Rulare script (opțional)
```bash
python -m pip install -r requirements.txt
python -m gdown --id 1D-Ua952YzK95yPCJxzr8SWu7ZJGVROJd -O weatherAUS.csv
python lab3_complete.py weatherAUS.csv lab3_results
```

## Metodologie
- Date: Rain in Australia / `weatherAUS.csv`.
- Split stratificat 70% train, 15% validare, 15% test, seed = 1.
- Tratarea NaN și standardizarea se învață numai pe train.
- Over-sampling / under-sampling se aplică doar pe train.
- Hiperparametrii se selectează pe validare, fără folosirea testului în selecție.
- Măsurătorile existente în PDF au fost efectuate pe CPU. Comparația GPU poate fi completată în Colab.

**Important:** nu încărca `weatherAUS.csv` în repository; datasetul este descărcat la rulare.
