# %% [markdown]
# # Environmental Adaptation Modeling (End-to-End Notebook)
# 
# This notebook mirrors the current codebase changes, including deterministic derived environment vectors.

# %% [markdown]
# ## 1) Setup

# %%
# Uncomment if needed
# !pip install torch transformers pandas numpy scikit-learn requests tqdm pyyaml

import json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None

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
print("[setup] imports ready, random seeds set, device:", DEVICE, flush=True)

# %% [markdown]
# ## 2) Config

# %%
CFG = {
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "output_dir": "outputs",
    "max_records": 20000,
    "extremophile_per_organism": 60,
    "n_simulated": 2000,
    "extreme_holdout_fraction": 0.2,
    "max_sequence_len": 256,
    "batch_size": 16,
    "num_epochs": 100,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "function_classes": 8,
    "env_dim": 10,
    "use_esm2": True,
    "esm2_model_name": "facebook/esm2_t6_8M_UR50D",
    "freeze_esm2": True,
    "esm2_batch_tokens": 1024,
}

for p in [CFG["raw_dir"], CFG["processed_dir"], CFG["output_dir"]]:
    Path(p).mkdir(parents=True, exist_ok=True)

print("[config] directories are ready:", CFG["raw_dir"], CFG["processed_dir"], CFG["output_dir"], flush=True)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_PAGE_SIZE = 250
NASA_EXO_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# %% [markdown]
# ## 3) Extremophile catalog + deterministic vector builders

# %%
EXTREMOPHILE_CATALOG = [
    {"organism": "Pyrococcus furiosus", "habitat": "hydrothermal_vent", "temperature_c": 100, "ph": 6.5, "salinity_psu": 35, "sulfur_rich": 0.95, "iron_rich": 0.85, "radiation_index": 0.2, "function_label": 6},
    {"organism": "Thermus aquaticus", "habitat": "hot_spring", "temperature_c": 75, "ph": 7.5, "salinity_psu": 3, "sulfur_rich": 0.55, "iron_rich": 0.45, "radiation_index": 0.2, "function_label": 5},
    {"organism": "Halobacterium salinarum", "habitat": "hypersaline_lake", "temperature_c": 42, "ph": 7.2, "salinity_psu": 260, "sulfur_rich": 0.35, "iron_rich": 0.30, "radiation_index": 0.4, "function_label": 4},
    {"organism": "Acidithiobacillus ferrooxidans", "habitat": "acid_mine_drainage", "temperature_c": 35, "ph": 2.0, "salinity_psu": 8, "sulfur_rich": 0.90, "iron_rich": 0.95, "radiation_index": 0.3, "function_label": 7},
    {"organism": "Deinococcus radiodurans", "habitat": "radiation_exposed_soil", "temperature_c": 32, "ph": 7.0, "salinity_psu": 1, "sulfur_rich": 0.25, "iron_rich": 0.25, "radiation_index": 0.95, "function_label": 3},
]

def _clip(x):
    return float(np.clip(x, 0.0, 1.0))

def derive_env_vector_from_exoplanet(row):
    teff = float(row.get("st_teff", 5500.0)) if pd.notna(row.get("st_teff")) else 5500.0
    met = float(row.get("st_met", 0.0)) if pd.notna(row.get("st_met")) else 0.0
    smass = float(row.get("st_mass", 1.0)) if pd.notna(row.get("st_mass")) else 1.0
    orbper = float(row.get("pl_orbper", 365.0)) if pd.notna(row.get("pl_orbper")) else 365.0
    rade = float(row.get("pl_rade", 1.0)) if pd.notna(row.get("pl_rade")) else 1.0

    temp_norm = _clip((teff - 2600.0) / 8000.0)
    metal_norm = _clip((met + 1.0) / 2.0)
    mass_norm = _clip((smass - 0.08) / 2.5)
    orbper_norm = _clip(np.log10(max(orbper, 0.1)) / 3.5)
    rade_norm = _clip((rade - 0.3) / 14.0)

    carbon = _clip(0.45 + 0.35 * metal_norm)
    nitrogen = _clip(0.40 + 0.25 * mass_norm)
    oxygen = _clip(0.35 + 0.40 * temp_norm)
    sulfur = _clip(0.25 + 0.30 * metal_norm + 0.10 * (1 - orbper_norm))
    phosphorus = _clip(0.30 + 0.20 * mass_norm)
    silicon = _clip(0.28 + 0.35 * rade_norm)
    iron = _clip(0.30 + 0.55 * metal_norm)
    magnesium = _clip(0.30 + 0.30 * metal_norm + 0.10 * mass_norm)
    sodium = _clip(0.25 + 0.25 * (1 - orbper_norm) + 0.10 * rade_norm)
    potassium = _clip(0.22 + 0.20 * (1 - orbper_norm) + 0.10 * mass_norm)
    return [carbon, nitrogen, oxygen, sulfur, phosphorus, silicon, iron, magnesium, sodium, potassium]

def derive_env_vector_from_extremophile_meta(meta):
    temp = float(meta.get("temperature_c", 37.0))
    ph = float(meta.get("ph", 7.0))
    salinity = float(meta.get("salinity_psu", 35.0))
    sulfur_rich = float(meta.get("sulfur_rich", 0.3))
    iron_rich = float(meta.get("iron_rich", 0.3))
    radiation = float(meta.get("radiation_index", 0.2))

    temp_norm = _clip((temp - 0.0) / 120.0)
    acidity = _clip((7.0 - ph) / 7.0)
    alkalinity = _clip((ph - 7.0) / 7.0)
    salinity_norm = _clip(salinity / 300.0)

    carbon = _clip(0.35 + 0.15 * temp_norm + 0.10 * (1 - salinity_norm))
    nitrogen = _clip(0.30 + 0.20 * temp_norm + 0.10 * radiation)
    oxygen = _clip(0.45 - 0.20 * temp_norm + 0.10 * alkalinity)
    sulfur = _clip(0.20 + 0.65 * sulfur_rich + 0.10 * acidity)
    phosphorus = _clip(0.30 + 0.10 * temp_norm + 0.05 * salinity_norm)
    silicon = _clip(0.25 + 0.20 * temp_norm + 0.05 * alkalinity)
    iron = _clip(0.20 + 0.70 * iron_rich + 0.10 * acidity)
    magnesium = _clip(0.25 + 0.35 * salinity_norm)
    sodium = _clip(0.20 + 0.75 * salinity_norm)
    potassium = _clip(0.20 + 0.50 * salinity_norm)
    return [carbon, nitrogen, oxygen, sulfur, phosphorus, silicon, iron, magnesium, sodium, potassium]

# %% [markdown]
# ## 4) Download data

# %%
import time

def safe_get(url, params=None, timeout=60, attempts=4, backoff=2):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "nyu-ai-in-genomics/1.0"},
            )
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == attempts:
                break
            wait_seconds = backoff * attempt
            print(f"[download] retrying {url} after {type(exc).__name__}: {exc} (attempt {attempt}/{attempts})", flush=True)
            time.sleep(wait_seconds)

    if last_exc is not None:
        raise last_exc

def fetch_uniprot_tsv(query, fields, max_records, page_size=UNIPROT_PAGE_SIZE):
    lines = []
    next_url = None
    fetched = 0
    page = 1

    while fetched < max_records:
        if next_url is None:
            params = {
                "query": query,
                "format": "tsv",
                "fields": fields,
                "size": min(page_size, max_records - fetched),
            }
            response = safe_get(UNIPROT_URL, params=params)
        else:
            response = safe_get(next_url)

        page_lines = response.text.strip().splitlines()
        if not page_lines:
            break

        if not lines:
            lines.extend(page_lines)
        else:
            lines.extend(page_lines[1:])

        fetched = max(0, len(lines) - 1)
        print(f"[download] fetched UniProt page {page} with {max(0, len(page_lines) - 1)} rows", flush=True)
        page += 1

        next_url = response.links.get("next", {}).get("url")
        if not next_url:
            break

    return lines[: max_records + 1] if lines else lines

def download_all(cfg):
    out_dir = Path(cfg["raw_dir"])
    print("[download] starting raw data fetch", flush=True)
    try:
        lines = fetch_uniprot_tsv(
            query="reviewed:true",
            fields="accession,sequence,protein_name,organism_name",
            max_records=cfg["max_records"],
        )
        (out_dir / "uniprot_sample.tsv").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print("[download] wrote uniprot_sample.tsv", flush=True)
    except Exception as e:
        print("[warn] uniprot sample failed", e, flush=True)

    try:
        query = f"select top {cfg['max_records']} pl_name,st_teff,st_met,st_mass,pl_orbper,pl_rade from pscomppars"
        (out_dir / "nasa_exoplanets.csv").write_text(safe_get(NASA_EXO_URL, params={"query":query,"format":"csv"}).text, encoding="utf-8")
        print("[download] wrote nasa_exoplanets.csv", flush=True)
    except Exception as e:
        print("[warn] nasa exoplanets failed", e, flush=True)

    ext_path = out_dir / "extremophile_sequences.jsonl"
    records=[]
    for item in EXTREMOPHILE_CATALOG:
        query = f'reviewed:true AND organism_name:"{item["organism"]}"'
        try:
            lines = fetch_uniprot_tsv(
                query=query,
                fields="accession,sequence,protein_name,organism_name",
                max_records=cfg["extremophile_per_organism"],
            )
            for row in lines[1:]:
                cols=row.split("\t")
                if len(cols)<4: continue
                records.append({"accession":cols[0],"sequence":cols[1],"protein_name":cols[2],"organism_name":cols[3],"habitat":item["habitat"],"env_vector":derive_env_vector_from_extremophile_meta(item),"function_label":item["function_label"]})
        except Exception:
            pass

    if not records:
        print("[download] no extremophile rows returned, using fallback records", flush=True)
        for i,item in enumerate(EXTREMOPHILE_CATALOG):
            records.append({"accession":f"fallback_{i}","sequence":"M"*60+"G"*40,"protein_name":"fallback","organism_name":item["organism"],"habitat":item["habitat"],"env_vector":derive_env_vector_from_extremophile_meta(item),"function_label":item["function_label"]})

    with ext_path.open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r)+"\n")
    print("[download] wrote extremophile_sequences.jsonl with", len(records), "records", flush=True)

    return


download_all(CFG)

# %% [markdown]
# ## 5) Build datasets (derived vectors)

# %%
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

def sequence_from_env(length, env_vec):
    probs = np.ones(len(AA), dtype=float)
    sulfur, iron, oxygen, salinity_proxy = float(env_vec[3]), float(env_vec[6]), float(env_vec[2]), float(env_vec[8])
    for aa in ["C","M","H"]: probs[np.where(AA==aa)[0][0]] *= 1.0 + sulfur + iron
    for aa in ["D","E"]: probs[np.where(AA==aa)[0][0]] *= 1.0 + salinity_proxy
    for aa in ["G","A"]: probs[np.where(AA==aa)[0][0]] *= 1.0 + oxygen*0.5
    probs /= probs.sum()
    return "".join(np.random.choice(AA, size=length, p=probs))

def build_datasets(cfg):
    raw_dir=Path(cfg["raw_dir"])
    out_train=Path(cfg["processed_dir"]) / "dataset.jsonl"
    out_eval=Path(cfg["processed_dir"]) / "extremophile_eval.jsonl"

    print("[build] starting dataset construction", flush=True)
    exo_path = raw_dir / "nasa_exoplanets.csv"
    exo_df = pd.read_csv(exo_path) if exo_path.exists() else pd.DataFrame([{} for _ in range(cfg["n_simulated"])])

    records=[]
    for i in range(cfg["n_simulated"]):
        row = exo_df.iloc[i % len(exo_df)] if len(exo_df)>0 else pd.Series(dtype=float)
        env = np.array(derive_env_vector_from_exoplanet(row))
        seq = sequence_from_env(np.random.randint(80,220), env)
        fn = int(np.digitize(env.mean(), bins=np.linspace(0.2,0.9,7)))
        records.append({"id":f"sim_{i}","env_vector":env.tolist(),"sequence":seq,"function_label":min(fn,7),"source":"simulated"})

    ext_rows=[]
    ext_file = raw_dir / "extremophile_sequences.jsonl"
    if ext_file.exists():
        with ext_file.open("r", encoding="utf-8") as f:
            for i,line in enumerate(f):
                r=json.loads(line)
                ext_rows.append({"id":f"ext_{i}","env_vector":r["env_vector"],"sequence":r["sequence"],"function_label":int(r.get("function_label",0)),"source":"extremophile"})

    np.random.shuffle(ext_rows)
    n_holdout=max(1,int(len(ext_rows)*cfg["extreme_holdout_fraction"])) if ext_rows else 0
    ext_eval=ext_rows[:n_holdout]; ext_train=ext_rows[n_holdout:]
    records.extend(ext_train)

    with out_train.open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r)+"\n")
    with out_eval.open("w", encoding="utf-8") as f:
        for r in ext_eval: f.write(json.dumps(r)+"\n")

    print("[build] train records:", len(records), "ext_eval records:", len(ext_eval), flush=True)

build_datasets(CFG)

# %% [markdown]
# ## 6) Tokenizer, dataset, model

# %%
AA_VOCAB="ACDEFGHIKLMNPQRSTVWYXBZUO"; SPECIAL=["<pad>","<bos>","<eos>"]
class SequenceTokenizer:
    def __init__(self,vocab=AA_VOCAB):
        self.itos=SPECIAL+list(vocab); self.stoi={t:i for i,t in enumerate(self.itos)}
        self.pad_id=self.stoi["<pad>"]; self.bos_id=self.stoi["<bos>"]; self.eos_id=self.stoi["<eos>"]
    @property
    def vocab_size(self): return len(self.itos)
    def encode(self,seq,max_len):
        ids=[self.bos_id]+[self.stoi.get(ch,self.stoi["X"]) for ch in seq[:max_len-2]]+[self.eos_id]
        ids += [self.pad_id]*(max_len-len(ids)); return ids

class EnvSequenceDataset(Dataset):
    def __init__(self,rows,tok,max_len): self.rows=rows; self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        r=self.rows[idx]; seq=torch.tensor(self.tok.encode(r["sequence"],self.max_len),dtype=torch.long)
        return {"env":torch.tensor(r["env_vector"],dtype=torch.float32),"x":seq[:-1],"y":seq[1:],"fn_label":torch.tensor(r["function_label"],dtype=torch.long)}

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=512):
        super().__init__(); pe=torch.zeros(max_len,d_model); pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class EnvConditionedGenerator(nn.Module):
    def __init__(self,vocab_size,env_dim,d_model,nhead,num_layers,dim_feedforward,dropout,max_len,function_classes):
        super().__init__(); self.token_emb=nn.Embedding(vocab_size,d_model); self.env_proj=nn.Linear(env_dim,d_model); self.pos=PositionalEncoding(d_model,max_len)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.decoder=nn.TransformerEncoder(layer,num_layers=num_layers); self.lm_head=nn.Linear(d_model,vocab_size)
        self.fn_head=nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model,function_classes))
    def forward(self,tokens,env):
        h=self.pos(self.token_emb(tokens)+self.env_proj(env).unsqueeze(1)); h=self.decoder(h)
        return self.lm_head(h), self.fn_head(h.mean(dim=1))

class ESM2ConditionedGenerator(nn.Module):
    def __init__(self,vocab_size,env_dim,esm2_model_name,d_model,nhead,num_layers,dim_feedforward,dropout,function_classes,freeze_esm2=True):
        super().__init__()
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("transformers is required for ESM2 backend")
        self.esm_tokenizer=AutoTokenizer.from_pretrained(esm2_model_name)
        self.esm_model=AutoModel.from_pretrained(esm2_model_name)
        self.esm_dim=self.esm_model.config.hidden_size
        if freeze_esm2:
            for p in self.esm_model.parameters(): p.requires_grad=False
        self.env_proj=nn.Linear(env_dim,self.esm_dim)
        layer=nn.TransformerEncoderLayer(d_model=self.esm_dim,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.decoder=nn.TransformerEncoder(layer,num_layers=num_layers)
        self.lm_head=nn.Linear(self.esm_dim,vocab_size)
        self.fn_head=nn.Sequential(nn.Linear(self.esm_dim,d_model),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model,function_classes))

    def _decode_to_strings(self,tokens,tok):
        seqs=[]
        for row in tokens.tolist():
            aa=[tok.itos[i] for i in row if i>=3]
            seq="".join(ch for ch in aa if len(ch)==1)
            seqs.append(seq if len(seq)>0 else "A")
        return seqs

    def forward(self,tokens,env,tok=None):
        if tok is None:
            raise ValueError("tok must be provided for ESM2ConditionedGenerator.forward")
        seqs=self._decode_to_strings(tokens,tok)
        esm_batch=self.esm_tokenizer(seqs,return_tensors="pt",padding=True,truncation=True,max_length=tokens.size(1))
        esm_batch={k:v.to(tokens.device) for k,v in esm_batch.items()}
        hidden=self.esm_model(**esm_batch).last_hidden_state
        if hidden.size(1)<tokens.size(1):
            pad=torch.zeros(hidden.size(0),tokens.size(1)-hidden.size(1),hidden.size(2),device=hidden.device)
            hidden=torch.cat([hidden,pad],dim=1)
        h=hidden[:,:tokens.size(1),:] + self.env_proj(env).unsqueeze(1)
        h=self.decoder(h)
        return self.lm_head(h), self.fn_head(h.mean(dim=1))


def build_model(cfg, tok):
    if cfg.get("use_esm2", False):
        print(f"[model] using ESM2 backend: {cfg['esm2_model_name']}", flush=True)
        return ESM2ConditionedGenerator(vocab_size=tok.vocab_size, env_dim=cfg["env_dim"], esm2_model_name=cfg["esm2_model_name"], d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], function_classes=cfg["function_classes"], freeze_esm2=cfg.get("freeze_esm2", True))
    print("[model] using baseline transformer backend", flush=True)
    return EnvConditionedGenerator(vocab_size=tok.vocab_size, env_dim=cfg["env_dim"], d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], max_len=cfg["max_sequence_len"], function_classes=cfg["function_classes"])


# %% [markdown]
# ## 7) Train

# %%
def load_jsonl(path):
    out=[]
    with open(path) as f:
        for line in f: out.append(json.loads(line))
    return out

print("[train] loading dataset jsonl", flush=True)
rows=load_jsonl(str(Path(CFG["processed_dir"])/"dataset.jsonl"))
print("[train] loaded rows:", len(rows), flush=True)
train_rows,temp_rows=train_test_split(rows,test_size=0.2,random_state=SEED)
val_rows,test_rows=train_test_split(temp_rows,test_size=0.5,random_state=SEED)

tok=SequenceTokenizer()
train_loader=DataLoader(EnvSequenceDataset(train_rows,tok,CFG["max_sequence_len"]),batch_size=CFG["batch_size"],shuffle=True)
val_loader=DataLoader(EnvSequenceDataset(val_rows,tok,CFG["max_sequence_len"]),batch_size=CFG["batch_size"],shuffle=False)

model=build_model(CFG, tok).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=CFG["lr"],weight_decay=CFG["weight_decay"])
ce=nn.CrossEntropyLoss(ignore_index=tok.pad_id); clf=nn.CrossEntropyLoss()

def run_epoch(loader, training=True):
    model.train(training); total=0.0; corr=0; den=0
    for b in tqdm(loader, leave=False):
        x=b["x"].to(DEVICE); y=b["y"].to(DEVICE); env=b["env"].to(DEVICE); fn=b["fn_label"].to(DEVICE)
        logits, fn_logits = model(x, env, tok) if CFG.get("use_esm2", False) else model(x, env)
        loss = ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) + 0.3*clf(fn_logits, fn)
        if training:
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total += loss.item(); corr += (fn_logits.argmax(-1)==fn).sum().item(); den += fn.size(0)
    return total/max(1,len(loader)), corr/max(1,den)

best=float("inf")
print("[train] starting training for", CFG["num_epochs"], "epochs", flush=True)
for ep in range(CFG["num_epochs"]):
    print(f"[train] epoch {ep + 1}/{CFG['num_epochs']} starting", flush=True)
    trl,tra = run_epoch(train_loader,True)
    with torch.no_grad(): val,vala = run_epoch(val_loader,False)
    print(ep+1, trl, val, vala, flush=True)
    if val<best: best=val; torch.save(model.state_dict(), str(Path(CFG["output_dir"])/"best_model.pt")); print("[train] saved new best checkpoint", flush=True)

# %% [markdown]
# ## 8) Extremophile eval

# %%
ext=load_jsonl(str(Path(CFG["processed_dir"])/"extremophile_eval.jsonl"))
if not ext: print("[eval] no extremophile eval rows", flush=True)
else:
    print("[eval] loading best checkpoint", flush=True)
    model.load_state_dict(torch.load(str(Path(CFG["output_dir"])/"best_model.pt"), map_location=DEVICE))
    loader=DataLoader(EnvSequenceDataset(ext,tok,CFG["max_sequence_len"]),batch_size=CFG["batch_size"],shuffle=False)
    model.eval(); lm=[]; fl=[]; c=0; n=0
    with torch.no_grad():
        for b in loader:
            x=b["x"].to(DEVICE); y=b["y"].to(DEVICE); env=b["env"].to(DEVICE); fn=b["fn_label"].to(DEVICE)
            logits, fn_logits=(model(x, env, tok) if CFG.get("use_esm2", False) else model(x, env))
            lm.append(ce(logits.reshape(-1,logits.size(-1)), y.reshape(-1)).item())
            fl.append(clf(fn_logits,fn).item())
            c += (fn_logits.argmax(-1)==fn).sum().item(); n += fn.size(0)
    print(json.dumps({"extremophile_lm_loss":float(sum(lm)/max(1,len(lm))),"extremophile_function_loss":float(sum(fl)/max(1,len(fl))),"extremophile_function_acc":float(c/max(1,n)),"n_eval_samples":n}, indent=2), flush=True)

# %% [markdown]
# ## 9) Sample alien condition

# %%
def apply_top_k(logits, k=50):
    """Keep only top-k logits, set others to -inf."""
    if k >= logits.shape[-1]:
        return logits
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    min_val = top_k_logits[..., -1:].clone()
    logits_filtered = logits.clone()
    logits_filtered[logits_filtered < min_val] = float('-inf')
    return logits_filtered

def apply_top_p(probs, p=0.9):
    """Keep tokens until cumulative probability >= p (nucleus sampling)."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum > p
    sorted_indices_to_remove[..., 0] = False  # keep at least one token
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs_filtered = probs.clone()
    probs_filtered[indices_to_remove] = 0.0
    return probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

def sample_sequence(env, max_len=256, temperature=1.0, top_k=None, top_p=None):
    """
    Generate amino acid sequence with optional top-k or top-p sampling.
    Args:
        env: environment vector
        max_len: maximum sequence length
        temperature: softmax temperature (higher = more diverse)
        top_k: keep only top-k tokens (None = disabled)
        top_p: nucleus sampling threshold (None = disabled)
    """
    print(f"[sample] generating sequence with temp={temperature}, top_k={top_k}, top_p={top_p}", flush=True)
    model.eval(); t=torch.tensor([[tok.bos_id]],dtype=torch.long,device=DEVICE); env_t=torch.tensor([env],dtype=torch.float32,device=DEVICE)
    with torch.no_grad():
        for _ in range(max_len-1):
            logits, fn_logits = (model(t, env_t, tok) if CFG.get("use_esm2", False) else model(t, env_t))
            logits_scaled = logits[:,-1,:]/max(temperature,1e-5)
            
            # Apply top-k filtering if specified
            if top_k is not None and top_k > 0:
                logits_scaled = apply_top_k(logits_scaled, k=top_k)
            
            probs = torch.softmax(logits_scaled, dim=-1)
            
            # Apply top-p filtering if specified
            if top_p is not None and 0 < top_p < 1.0:
                probs = apply_top_p(probs, p=top_p)
            
            nxt=torch.multinomial(probs,1); t=torch.cat([t,nxt],dim=1)
            if nxt.item()==tok.eos_id: break
    seq=''.join(tok.itos[i] for i in t[0].tolist() if i>=3)
    print(f"[sample] generation complete, length: {len(seq)}", flush=True)
    return seq, int(fn_logits.argmax(-1).item())

# Example 1: vanilla sampling (baseline)
print("\n[example] Sampling with vanilla temperature strategy", flush=True)
env=[0.7,0.5,0.6,0.8,0.4,0.3,0.9,0.4,0.2,0.5]
seq,fn=sample_sequence(env, max_len=CFG["max_sequence_len"], temperature=1.0)
print(json.dumps({"strategy":"vanilla_temp_1.0","environment":env,"generated_sequence":seq[:200],"predicted_function_class":fn}, indent=2), flush=True)

# Example 2: higher temperature for diversity
print("\n[example] Sampling with higher temperature (1.5)", flush=True)
seq,fn=sample_sequence(env, max_len=CFG["max_sequence_len"], temperature=1.5)
print(json.dumps({"strategy":"higher_temp_1.5","environment":env,"generated_sequence":seq[:200],"predicted_function_class":fn}, indent=2), flush=True)

# Example 3: top-k sampling (keep top 50 tokens)
print("\n[example] Sampling with top-k=50", flush=True)
seq,fn=sample_sequence(env, max_len=CFG["max_sequence_len"], temperature=1.0, top_k=50)
print(json.dumps({"strategy":"top_k_50","environment":env,"generated_sequence":seq[:200],"predicted_function_class":fn}, indent=2), flush=True)

# Example 4: top-p (nucleus) sampling with p=0.9
print("\n[example] Sampling with top-p=0.9", flush=True)
seq,fn=sample_sequence(env, max_len=CFG["max_sequence_len"], temperature=1.0, top_p=0.9)
print(json.dumps({"strategy":"top_p_0.9","environment":env,"generated_sequence":seq[:200],"predicted_function_class":fn}, indent=2), flush=True)


