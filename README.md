# AI in Genomics Project

This repository builds an environmental protein generation pipeline from GOLD, UniProt, NASA POWER, SoilGrids, and WorldClim data.

## Workflow

Run the steps in this order:

1. Generate the combined environmental parquet file.

```bash
python src/main.py
```

2. Clean the parquet and produce the CSV you want.

```bash
python src/clean_data.py \
  --input-parquet data/combined_uniprot_and_gold_environmental_data_10K.parquet \
  --output-csv combined_uniprot_and_gold_environmental_data_10K.csv \
  --cleaned-output-csv cleaned_environmental_data.csv
```

3. Train the model and create the checkpoint.

```bash
python src/train.py \
  --data cleaned_environmental_data.csv \
  --checkpoint checkpoints/best_alien_protein_model.pt
```

4. Launch the Streamlit UI using the checkpoint and scaler.

```bash
streamlit run src/app.py -- \
  --checkpoint checkpoints/best_alien_protein_model.pt \
  --scaler checkpoints/environmental_scaler.pkl
```

## Notes

- `src/main.py` combines GOLD rows with UniProt sequences and enriches them with NASA POWER, SoilGrids, and WorldClim features.
- `src/clean_data.py` removes corrupted numeric rows from the parquet-derived dataset.
- `src/train.py` trains the conditional protein model and saves both the scaler and checkpoint.
- `src/app.py` loads the checkpoint and scaler to generate sequences in the UI.
