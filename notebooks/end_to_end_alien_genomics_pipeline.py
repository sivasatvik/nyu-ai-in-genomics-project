# %% [markdown]
# # Environmental Adaptation Modeling (End-to-End Notebook)
# 
# Run this notebook top-to-bottom to:
# 1. Download data (UniProt + NASA + Earth extremophiles)
# 2. Build train + extremophile holdout datasets
# 3. Train an environment-conditioned sequence model
# 4. Evaluate on extremophile holdout
# 5. Sample alien-environment sequences

# %% [markdown]
# ## 1) Setup

# %%
# If needed, uncomment:
# !pip install torch transformers pandas numpy scikit-learn requests tqdm pyyaml

import os, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
# Force line-buffered stdout so print statements appear immediately in sbatch /
# Singularity logs rather than being flushed only at script exit.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


# %% [markdown]
# ## 2) Configuration

# %%
CFG = {
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "output_dir": "outputs",
    "max_records": 5000,
    "extremophile_per_organism": 60,
    "n_simulated": 2000,
    "extreme_holdout_fraction": 0.2,
    "max_sequence_len": 256,
    "batch_size": 16,
    "num_epochs": 5,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "function_classes": 8,
    "env_dim": 10,
}

for p in [CFG["raw_dir"], CFG["processed_dir"], CFG["output_dir"]]:
    Path(p).mkdir(parents=True, exist_ok=True)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
NASA_EXO_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


# %% [markdown]
# ## 3) Extremophile Catalog

# %%
EXTREMOPHILE_CATALOG = [
    {"organism": "Pyrococcus furiosus", "habitat": "hydrothermal_vent", "env_vector": [0.45,0.35,0.15,0.85,0.30,0.20,0.80,0.60,0.50,0.40], "function_label": 6},
    {"organism": "Thermus aquaticus", "habitat": "hot_spring", "env_vector": [0.50,0.40,0.30,0.55,0.35,0.25,0.50,0.45,0.30,0.35], "function_label": 5},
    {"organism": "Halobacterium salinarum", "habitat": "hypersaline_lake", "env_vector": [0.55,0.40,0.60,0.45,0.30,0.20,0.35,0.40,0.95,0.70], "function_label": 4},
    {"organism": "Acidithiobacillus ferrooxidans", "habitat": "acid_mine_drainage", "env_vector": [0.40,0.35,0.40,0.80,0.30,0.20,0.90,0.45,0.25,0.30], "function_label": 7},
    {"organism": "Deinococcus radiodurans", "habitat": "radiation_exposed_soil", "env_vector": [0.50,0.45,0.55,0.40,0.35,0.20,0.35,0.35,0.30,0.35], "function_label": 3},
]


# %% [markdown]
# ## 4) Data Download (UniProt, NASA, Extremophiles)

# %%
def safe_get(url, params=None, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r

def download_all(cfg):
    out_dir = Path(cfg["raw_dir"])

    # 1) UniProt general sample
    try:
        params = {"query":"reviewed:true","format":"tsv","fields":"accession,sequence,protein_name,organism_name","size":cfg["max_records"]}
        (out_dir / "uniprot_sample.tsv").write_text(safe_get(UNIPROT_URL, params=params).text, encoding="utf-8")
        print("[ok] uniprot_sample.tsv")
    except Exception as e:
        print("[warn] uniprot sample failed:", e)

    # 2) NASA exoplanets
    try:
        query = f"select top {cfg['max_records']} pl_name,st_teff,st_met,st_mass,pl_orbper,pl_rade from pscomppars"
        (out_dir / "nasa_exoplanets.csv").write_text(safe_get(NASA_EXO_URL, params={"query":query,"format":"csv"}).text, encoding="utf-8")
        print("[ok] nasa_exoplanets.csv")
    except Exception as e:
        print("[warn] nasa exoplanets failed:", e)

    # 3) Extremophile sequences
    ext_path = out_dir / "extremophile_sequences.jsonl"
    records = []
    for item in EXTREMOPHILE_CATALOG:
        params = {
            "query": f"reviewed:true AND organism_name:\"{item['organism']}\"",
            "format": "tsv",
            "fields": "accession,sequence,protein_name,organism_name",
            "size": cfg["extremophile_per_organism"],
        }
        try:
            lines = safe_get(UNIPROT_URL, params=params).text.strip().splitlines()
            for row in lines[1:]:
                cols = row.split("\t")
                if len(cols) < 4:
                    continue
                records.append({
                    "accession": cols[0], "sequence": cols[1], "protein_name": cols[2], "organism_name": cols[3],
                    "habitat": item["habitat"], "env_vector": item["env_vector"], "function_label": item["function_label"]
                })
        except Exception:
            pass

    if not records:
        for i, item in enumerate(EXTREMOPHILE_CATALOG):
            records.append({
                "accession": f"fallback_{i}",
                "sequence": "M"*60 + "G"*40,
                "protein_name": "fallback_protein",
                "organism_name": item["organism"],
                "habitat": item["habitat"],
                "env_vector": item["env_vector"],
                "function_label": item["function_label"],
            })

    with ext_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[ok] extremophile_sequences.jsonl ({len(records)} rows)")

download_all(CFG)


# %% [markdown]
# ## 5) Build Processed Datasets (train + extremophile holdout)

# %%
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

def random_sequence(length, env_vec):
    probs = np.ones(len(AA), dtype=float)
    sulfur = float(env_vec[3]) if len(env_vec)>3 else 0.1
    iron = float(env_vec[6]) if len(env_vec)>6 else 0.1
    bias = min(3.0, 1.0 + sulfur + iron)
    for aa in ["C","M","H"]:
        probs[np.where(AA == aa)[0][0]] *= bias
    probs /= probs.sum()
    return "".join(np.random.choice(AA, size=length, p=probs))

def construct_env_vector(row):
    base = np.clip(np.random.normal(0.5, 0.2, size=10), 0.0, 1.0)
    if "st_met" in row and pd.notna(row.get("st_met")): base[6] = float(np.clip((row["st_met"]+1.0)/2.0,0.0,1.0))
    if "st_teff" in row and pd.notna(row.get("st_teff")): base[2] = float(np.clip((row["st_teff"]-2500)/7000,0.0,1.0))
    return base.tolist()

def build_datasets(cfg):
    raw_dir = Path(cfg["raw_dir"])
    out_train = Path(cfg["processed_dir"]) / "dataset.jsonl"
    out_ext_eval = Path(cfg["processed_dir"]) / "extremophile_eval.jsonl"

    exo_path = raw_dir / "nasa_exoplanets.csv"
    exo_df = pd.read_csv(exo_path) if exo_path.exists() else pd.DataFrame([{} for _ in range(cfg["n_simulated"])])

    records = []
    for i in range(cfg["n_simulated"]):
        row = exo_df.iloc[i % len(exo_df)] if len(exo_df)>0 else pd.Series(dtype=float)
        env = np.array(construct_env_vector(row))
        seq = random_sequence(np.random.randint(80, 220), env)
        fn = int(np.digitize(env.mean(), bins=np.linspace(0.2,0.9,7)))
        records.append({"id":f"sim_{i}","env_vector":env.tolist(),"sequence":seq,"function_label":min(fn,7),"source":"simulated"})

    ext_rows = []
    ext_file = raw_dir / "extremophile_sequences.jsonl"
    if ext_file.exists():
        with ext_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                ext_rows.append({
                    "id":f"ext_{i}","env_vector":r["env_vector"],"sequence":r["sequence"],
                    "function_label":int(r.get("function_label",0)),"source":"extremophile",
                    "habitat":r.get("habitat","unknown"),"organism_name":r.get("organism_name","unknown")
                })

    np.random.shuffle(ext_rows)
    n_holdout = max(1, int(len(ext_rows) * cfg["extreme_holdout_fraction"])) if ext_rows else 0
    ext_eval = ext_rows[:n_holdout]
    ext_train = ext_rows[n_holdout:]
    records.extend(ext_train)

    with out_train.open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r) + "\n")

    with out_ext_eval.open("w", encoding="utf-8") as f:
        for r in ext_eval: f.write(json.dumps(r) + "\n")

    print(f"[ok] train rows={len(records)} -> {out_train}")
    print(f"[ok] extremophile holdout rows={len(ext_eval)} -> {out_ext_eval}")

build_datasets(CFG)


# %% [markdown]
# ## 6) Tokenizer, Dataset, Model

# %%
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYXBZUO"
SPECIAL = ["<pad>","<bos>","<eos>"]

class SequenceTokenizer:
    def __init__(self, vocab=AA_VOCAB):
        self.itos = SPECIAL + list(vocab)
        self.stoi = {t:i for i,t in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]; self.bos_id = self.stoi["<bos>"]; self.eos_id = self.stoi["<eos>"]
    @property
    def vocab_size(self): return len(self.itos)
    def encode(self, seq, max_len):
        ids = [self.bos_id] + [self.stoi.get(ch, self.stoi["X"]) for ch in seq[:max_len-2]] + [self.eos_id]
        ids += [self.pad_id]*(max_len-len(ids))
        return ids

class EnvSequenceDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows=rows; self.tok=tokenizer; self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r=self.rows[idx]
        seq=torch.tensor(self.tok.encode(r["sequence"], self.max_len), dtype=torch.long)
        x=seq[:-1]; y=seq[1:]
        return {"env":torch.tensor(r["env_vector"],dtype=torch.float32),"x":x,"y":y,"fn_label":torch.tensor(r["function_label"],dtype=torch.long)}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self,x): return x + self.pe[:, :x.size(1)]

class EnvConditionedGenerator(nn.Module):
    def __init__(self, vocab_size, env_dim, d_model, nhead, num_layers, dim_feedforward, dropout, max_len, function_classes):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size, d_model)
        self.env_proj=nn.Linear(env_dim, d_model)
        self.pos=PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.function_head = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model,function_classes))
    def forward(self, tokens, env):
        h = self.token_emb(tokens) + self.env_proj(env).unsqueeze(1)
        h = self.decoder(self.pos(h))
        return self.lm_head(h), self.function_head(h.mean(dim=1))


# %% [markdown]
# ## 7) Training

# %%
def load_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    return rows

rows = load_jsonl(str(Path(CFG["processed_dir"]) / "dataset.jsonl"))
train_rows, temp_rows = train_test_split(rows, test_size=0.2, random_state=SEED)
val_rows, test_rows = train_test_split(temp_rows, test_size=0.5, random_state=SEED)

tok = SequenceTokenizer()
train_ds = EnvSequenceDataset(train_rows, tok, CFG["max_sequence_len"])
val_ds = EnvSequenceDataset(val_rows, tok, CFG["max_sequence_len"])
test_ds = EnvSequenceDataset(test_rows, tok, CFG["max_sequence_len"])

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False)

model = EnvConditionedGenerator(
    vocab_size=tok.vocab_size,
    env_dim=CFG["env_dim"],
    d_model=CFG["d_model"],
    nhead=CFG["nhead"],
    num_layers=CFG["num_layers"],
    dim_feedforward=CFG["dim_feedforward"],
    dropout=CFG["dropout"],
    max_len=CFG["max_sequence_len"],
    function_classes=CFG["function_classes"],
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
ce = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
clf = nn.CrossEntropyLoss()

def run_epoch(loader, training=True):
    model.train(training)
    total=0.0; fnc=0; den=0
    for b in tqdm(loader, leave=False):
        x=b["x"].to(DEVICE); y=b["y"].to(DEVICE); env=b["env"].to(DEVICE); fn=b["fn_label"].to(DEVICE)
        logits, fn_logits = model(x, env)
        lm = ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        fl = clf(fn_logits, fn)
        loss = lm + 0.3*fl
        if training:
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item()
        fnc += (fn_logits.argmax(-1)==fn).sum().item(); den += fn.size(0)
    return total/max(1,len(loader)), fnc/max(1,den)

best = float("inf")
for epoch in range(CFG["num_epochs"]):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    with torch.no_grad():
        va_loss, va_acc = run_epoch(val_loader, False)
    print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_fn_acc={tr_acc:.3f} val_loss={va_loss:.4f} val_fn_acc={va_acc:.3f}")
    if va_loss < best:
        best = va_loss
        torch.save(model.state_dict(), str(Path(CFG["output_dir"]) / "best_model.pt"))

print("[ok] saved", Path(CFG["output_dir"]) / "best_model.pt")


# %% [markdown]
# ## 8) Extremophile Holdout Evaluation

# %%
ext_rows = load_jsonl(str(Path(CFG["processed_dir"]) / "extremophile_eval.jsonl"))
if len(ext_rows)==0:
    print("[warn] extremophile_eval.jsonl is empty")
else:
    model.load_state_dict(torch.load(str(Path(CFG["output_dir"]) / "best_model.pt"), map_location=DEVICE))
    ext_ds = EnvSequenceDataset(ext_rows, tok, CFG["max_sequence_len"])
    ext_loader = DataLoader(ext_ds, batch_size=CFG["batch_size"], shuffle=False)

    model.eval()
    lm_losses=[]; fn_losses=[]; corr=0; total=0
    with torch.no_grad():
        for b in ext_loader:
            x=b["x"].to(DEVICE); y=b["y"].to(DEVICE); env=b["env"].to(DEVICE); fn=b["fn_label"].to(DEVICE)
            logits, fn_logits = model(x, env)
            lm_losses.append(ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
            fn_losses.append(clf(fn_logits, fn).item())
            corr += (fn_logits.argmax(-1)==fn).sum().item(); total += fn.size(0)

    print(json.dumps({
        "extremophile_lm_loss": float(sum(lm_losses)/max(1,len(lm_losses))),
        "extremophile_function_loss": float(sum(fn_losses)/max(1,len(fn_losses))),
        "extremophile_function_acc": float(corr/max(1,total)),
        "n_eval_samples": total
    }, indent=2))


# %% [markdown]
# ## 9) Sample Sequence for Alien Environment

# %%
def sample_sequence(env, max_len=256, temperature=1.0):
    model.eval()
    tokens = torch.tensor([[tok.bos_id]], dtype=torch.long, device=DEVICE)
    env_t = torch.tensor([env], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_len-1):
            logits, fn_logits = model(tokens, env_t)
            probs = torch.softmax(logits[:, -1, :] / max(temperature, 1e-5), dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if nxt.item() == tok.eos_id:
                break
    ids = tokens[0].tolist()
    seq = "".join(tok.itos[i] for i in ids if i >= 3)
    fn_cls = int(fn_logits.argmax(-1).item())
    return seq, fn_cls

alien_env = [0.7,0.5,0.6,0.8,0.4,0.3,0.9,0.4,0.2,0.5]
seq, fn = sample_sequence(alien_env, max_len=CFG["max_sequence_len"])
print(json.dumps({"environment": alien_env, "generated_sequence": seq[:200], "predicted_function_class": fn}, indent=2))



