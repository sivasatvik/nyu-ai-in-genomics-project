# Environmental Adaptation Modeling (Notebook-First Workflow)

This project is designed to run from a **single notebook**:

- `notebooks/end_to_end_alien_genomics_pipeline.ipynb`

Run the notebook top-to-bottom to execute the full pipeline:
1. data download,
2. dataset construction,
3. model training,
4. extremophile holdout evaluation,
5. alien-environment sequence sampling.

---

## End-to-end workflow in the notebook

### Section 1 — Setup
Installs/imports required libraries and configures deterministic seeds.

What it does:
- Imports `torch`, `numpy`, `pandas`, `requests`, `scikit-learn`, `tqdm`.
- Sets random seeds for reproducibility.
- Detects GPU/CPU device.

### Section 2 — Configuration
Defines all runtime parameters in one cell (`CFG`), including:
- paths: `data/raw`, `data/processed`, `outputs`
- data scale: `max_records`, `n_simulated`, `extremophile_per_organism`
- model/training: sequence length, batch size, epochs, LR, transformer dims.

### Section 3 — Extremophile Catalog
Creates a curated Earth-extremophile set with:
- organism name,
- habitat type,
- normalized 10D environment vector,
- coarse function class label.

### Section 4 — Data Download
Downloads and writes:
- `data/raw/uniprot_sample.tsv`
- `data/raw/nasa_exoplanets.csv`
- `data/raw/extremophile_sequences.jsonl`

Behavior details:
- UniProt and NASA pulls are attempted via HTTP.
- Extremophile sequences are queried per curated organism.
- If downloads fail, fallback extremophile records are generated so the notebook remains runnable.

### Section 5 — Processed Dataset Build
Builds:
- training dataset: `data/processed/dataset.jsonl`
- extremophile holdout: `data/processed/extremophile_eval.jsonl`

What it does:
- creates simulated environment-conditioned sequences,
- merges extremophile records into training,
- holds out a fraction for extremophile validation,
- tags record source (`simulated` vs `extremophile`).

### Section 6 — Tokenizer, Dataset, Model
Defines:
- amino-acid tokenizer with special tokens (`<pad>`, `<bos>`, `<eos>`),
- PyTorch dataset wrapper,
- environment-conditioned transformer model with:
  - sequence generation head,
  - function classification head.

### Section 7 — Training
Runs train/validation loops and saves the best checkpoint:
- output model: `outputs/best_model.pt`

What it reports:
- training loss,
- validation loss,
- function classification accuracy.

### Section 8 — Extremophile Holdout Evaluation
Evaluates the trained model against Earth extremophile holdout examples.

Metrics reported:
- extremophile LM loss,
- extremophile function loss,
- extremophile function accuracy,
- number of eval samples.

### Section 9 — Alien Environment Sampling
Generates a sequence conditioned on a user-defined alien environment vector and predicts the function class.

Output:
- generated amino-acid sequence,
- predicted function class.

---

## How to run

1. Open `notebooks/end_to_end_alien_genomics_pipeline.ipynb` in Jupyter/VS Code.
2. Run cells in order from Section 1 through Section 9.
3. Confirm artifacts are created:
   - `data/raw/*`
   - `data/processed/dataset.jsonl`
   - `data/processed/extremophile_eval.jsonl`
   - `outputs/best_model.pt`

---

## Notes
- The notebook is the authoritative workflow for this project.
- Script files under `src/` are retained for modular reuse, but you can ignore them if you only want the single-notebook flow.
