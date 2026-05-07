# Environmental Adaptation Modeling

This project now supports:
1. **Derived environment vectors** (deterministic builder from source metadata).
2. **End-to-end notebook workflow** (`notebooks/end_to_end_alien_genomics_pipeline.ipynb`).
3. **Base web UI** for environment controls and amino-acid sequence generation (`app.py`).

---

## 1) Fully derived environment vector builder

Environment vectors are generated with explicit deterministic transforms in:
- `src/data/vector_builder.py`

Two derivation paths:
- `derive_env_vector_from_exoplanet(row)` from NASA columns (`st_teff`, `st_met`, `st_mass`, `pl_orbper`, `pl_rade`).
- `derive_env_vector_from_extremophile_meta(meta)` from extremophile metadata (`temperature_c`, `ph`, `salinity_psu`, sulfur/iron/radiation proxies).

Output schema (10D):
`[carbon, nitrogen, oxygen, sulfur, phosphorus, silicon, iron, magnesium, sodium, potassium]`

---

## 2) Notebook workflow (single run)

Use:
- `notebooks/end_to_end_alien_genomics_pipeline.ipynb`

Run top-to-bottom for:
- data download,
- dataset build,
- model training,
- extremophile holdout evaluation,
- alien condition sequence sampling.

---

## 3) CLI workflow

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download raw data
```bash
PYTHONPATH=src python src/data/download_data.py --out_dir data/raw --max_records 5000 --extremophile_per_organism 60
```

### Build datasets
```bash
PYTHONPATH=src python src/data/build_dataset.py --raw_dir data/raw --out_path data/processed/dataset.jsonl --extreme_eval_path data/processed/extremophile_eval.jsonl --n_samples 2000 --extreme_holdout_fraction 0.2
```

### Train
```bash
PYTHONPATH=src python src/train.py --config config.yaml --dataset data/processed/dataset.jsonl
```

### Evaluate on extremophile holdout
```bash
PYTHONPATH=src python src/extremophile_validate.py --config config.yaml --model_path outputs/best_model.pt --extreme_eval_path data/processed/extremophile_eval.jsonl
```

### Sample from CLI
```bash
PYTHONPATH=src python src/evaluate.py --config config.yaml --model_path outputs/best_model.pt --env "0.7,0.5,0.6,0.8,0.4,0.3,0.9,0.4,0.2,0.5"
```

---

## 4) Base Web UI

Run:
```bash
streamlit run app.py
```

In the UI you can:
- adjust each environment condition using sliders,
- set sampling temperature,
- generate a sequence from the saved model checkpoint,
- view predicted function class and output sequence.

> Ensure `outputs/best_model.pt` exists first (train the model before using UI inference).


Note: The notebook implements the same deterministic vector derivation logic used in `src/data/vector_builder.py`.
