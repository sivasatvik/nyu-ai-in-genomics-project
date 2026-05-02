# Proposal Alignment and Sequence Accuracy Review

## Scope reviewed
- Data generation and preprocessing pipeline (`src/data/*`)
- Conditional generator architecture and training (`src/models/*`, `src/train.py`)
- Inference/sampling behavior (`src/evaluate.py`, `app.py`)
- Holdout validation (`src/extremophile_validate.py`)

## High-level verdict
The repository has a coherent **end-to-end prototype** for conditioned sequence generation, but it is **not yet biologically reliable enough** for a proposal goal that implies accurate folding/functional plausibility.

In short:
- ✅ Good: deterministic environment-vector derivation, train/val/test flow, holdout eval plumbing.
- ⚠️ Gap: most training data are synthetic sequences made from amino-acid frequency heuristics rather than true structure-conditioned sequence data.
- ⚠️ Gap: no explicit folding/structure objective during training or post-generation filtering.

## What is correct and consistent

1. **Environment conditioning is consistently implemented**
   - Environment vectors are produced and attached to each record, then injected into the model via `env_proj` and added to token embeddings.

2. **Training objective is technically valid for language modeling + auxiliary classification**
   - LM cross-entropy and function-classification cross-entropy are combined in training.

3. **Sampling controls exist**
   - `temperature`, `top_k`, `top_p` are available in CLI inference.

4. **Extremophile holdout path exists**
   - The project has a dedicated extremophile evaluation script and held-out split logic.

## Why current outputs can still "hallucinate" sequences

1. **Synthetic target sequences dominate the dataset**
   - `sequence_from_env` samples amino acids from simple weighted frequencies, not from fold constraints, motif constraints, or sequence families.

2. **No structural supervision**
   - Training only sees token-level LM loss + coarse function label loss.
   - There is no loss term for secondary structure, contact consistency, pLDDT, or fold class agreement.

3. **No validity filter after generation**
   - Generated sequences are emitted directly without checks for:
     - composition/pathology (e.g., repeats, low complexity),
     - disorder propensity,
     - predicted fold confidence,
     - nearest-neighbor distance to known proteins.

4. **Function label construction is weak for simulated data**
   - For simulated rows, labels come from `digitize(env.mean())`, which can teach shortcut correlations unrelated to biochemical function.

## Concrete roadmap to generate more accurate/folding-plausible sequences

### Phase 1 (fastest quality jump): Add post-generation quality gates
For each generated candidate sequence:
1. Reject if invalid length/composition/low complexity.
2. Score using a protein language model perplexity proxy.
3. Run structure prediction (e.g., ESMFold/OpenFold/AlphaFold-compatible workflow).
4. Keep only candidates with high confidence (e.g., pLDDT threshold) and no severe clashes.
5. Optionally rank by environment-conditioned property proxy.

Generate N candidates per environment vector, keep top K by combined score.

### Phase 2: Improve training data realism
1. Increase real protein sequence proportion (extremophile + curated public protein sets).
2. Attach richer labels:
   - taxonomy/ecology tags,
   - domain annotations,
   - GO/EC or coarse function families.
3. Reduce reliance on heuristic synthetic sequences.
4. If synthetic generation is needed, use template- or motif-constrained synthesis instead of independent token sampling.

### Phase 3: Add structure-aware training signals
1. Multi-task heads for predicted secondary structure / solvent exposure / disorder.
2. Distillation signal from a frozen structure predictor or inverse folding model.
3. Contrastive objective: generated sequence embeddings should align with real proteins from similar environments.

### Phase 4: Controlled decoding instead of free sampling
1. Use beam search or diverse beam search with constraints.
2. Constrain disallowed motifs/composition on-the-fly.
3. Calibrate temperature per position or entropy-controlled decoding.

## Minimal implementation plan for this repo

1. Add a candidate generation script:
   - Inputs: env vector, `N`, decoding params.
   - Outputs: ranked candidates + diagnostics.

2. Add validators module:
   - Sequence heuristics (length, AA distribution, repeats).
   - Optional external-model hooks for fold confidence.

3. Extend evaluation:
   - Report acceptance rate after filters,
   - report average confidence metrics,
   - compare to baseline random/synthetic generation.

4. Update Streamlit UI:
   - "Generate 32 candidates" button,
   - show top 5 with confidence and rejection reasons.

## Acceptance criteria for "less hallucination"
A generation setup is improved only if it demonstrates:
1. Higher pass-rate on hard biochemical validity checks.
2. Better fold-confidence metrics on held-out test conditions.
3. Comparable or better diversity (non-collapsed outputs).
4. Stable performance on extremophile holdout evaluation.

## Practical note
Without a fold-aware model in the training or selection loop, sequence realism is expected to plateau even if sampling parameters are tuned.
