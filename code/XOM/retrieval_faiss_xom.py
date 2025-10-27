# ==== XOM OOD Retrieval Pipeline: builds z_retr and features (with causal restandardization) ====
# Outputs:
#   - /home/s6skkhan/macro_retrieval/XOM/test/xom_test_ood.parquet
#   - /home/s6skkhan/macro_retrieval/XOM/test/XOM_features.parquet
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------- Paths ----------
ROOT      = Path("/home/s6skkhan/macro_retrieval")
TRAIND    = ROOT / "train"              # shared across tickers
XOM_ROOT  = ROOT / "XOM"
TESTD     = XOM_ROOT / "test"
TESTD.mkdir(parents=True, exist_ok=True)

# Inputs (should already exist)
ood_base_path = TESTD / "xom_test_ood_base.parquet"      # created by earlier XOM base step

# Outputs
ood_out_path      = TESTD / "xom_test_ood.parquet"       # base + z_retr
features_out_path = TESTD / "XOM_features.parquet"       # final features (incl. *_rollz)

# Required train artifacts
index_path = TRAIND / "index_flat_ip.faiss"
meta_path  = TRAIND / "index_meta.parquet"
train_feat = TRAIND / "sp500_features.parquet"           # used for causal rolling stats & text fallback
text_mat   = TRAIND / "text_train.npy"                   # speeds up neighbor aggregation

# Retrieval knobs
alpha = 0.5   # macro weight
K     = 5
B     = 512

# ---------- Helpers ----------
def to_vec(v):
    """Convert value to float32 numpy vector (handles ndarray, list, bytes, json-str)."""
    if isinstance(v, np.ndarray): return v.astype("float32")
    if isinstance(v, list):       return np.asarray(v, dtype="float32")
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="strict")
    if isinstance(v, str):
        return np.asarray(json.loads(v), dtype="float32")
    raise TypeError(f"Unsupported embed type: {type(v)}")

def vec_to_list(v):
    """Convert value to a Python list[float] for Parquet (handles ndarray, list, bytes, json-str)."""
    if isinstance(v, list):       return v
    if isinstance(v, np.ndarray): return v.astype("float32").tolist()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="strict")
    if isinstance(v, str):
        return json.loads(v)
    raise TypeError(f"Unsupported vector type: {type(v)}")

def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)

# ---------- Preflight checks ----------
need = [index_path, meta_path, ood_base_path]
missing = [str(p) for p in need if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required file(s):\n" + "\n".join(" - "+m for m in missing))

# ---------- Load OOD base ----------
ood = pd.read_parquet(ood_base_path, engine="pyarrow").sort_values("Date").reset_index(drop=True)
print("Loaded XOM OOD base:", ood.shape, "|", ood["Date"].min().date(), "→", ood["Date"].max().date())

macro_cols = ["cpi_yoy_lagged_z","unrate_lagged_z","t10y2y_lagged_z","gdp_qoq_lagged_z"]
for c in macro_cols:
    if ood[c].dtype == "O":
        ood[c] = pd.to_numeric(ood[c], errors="coerce")

# Normalize embeddings to ndarray
embeds = ood["text_embed"].apply(to_vec)
print("Embed dim:", len(embeds.iloc[0]))

# ---------- Build joint queries ----------
text_q  = np.vstack(embeds.to_numpy()).astype("float32")           # [T, 384]
macro_q = ood[macro_cols].to_numpy(dtype="float32")                 # [T,   4]
joint_q = np.concatenate([text_q, alpha * macro_q], axis=1)         # [T, 388]
joint_q /= (np.linalg.norm(joint_q, axis=1, keepdims=True) + 1e-9)
print("Joint shape:", joint_q.shape)

# ---------- FAISS retrieval (GPU if available) ----------
import faiss
index_cpu = faiss.read_index(str(index_path))
index = index_cpu
used_gpu = False
try:
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        used_gpu = True
except Exception as e:
    print("GPU init failed; falling back to CPU. Reason:", repr(e))

meta = pd.read_parquet(meta_path, engine="pyarrow")["date"].to_numpy(dtype="datetime64[ns]")
print(f"FAISS ntotal={index.ntotal}, dim={index.d}, GPU={used_gpu} | meta dates:", meta.shape)

# Train text matrix for neighbor aggregation
if text_mat.exists():
    text_train = np.load(text_mat).astype("float32")
    print("Loaded text_train.npy:", text_train.shape)
else:
    tr = pd.read_parquet(train_feat, engine="pyarrow")
    text_train = np.vstack(tr["text_embed"].to_numpy()).astype("float32")
    print("Built text_train from sp500_features.parquet:", text_train.shape)

# ---------- Batched search with causal mask ----------
t0 = time.time()
dates_q = ood["Date"].to_numpy(dtype="datetime64[ns]")
z_retr = np.zeros_like(text_q, dtype="float32")
no_past_cnt, eff_k_hist = 0, []

for s in range(0, joint_q.shape[0], B):
    e = min(s + B, joint_q.shape[0])
    D, I = index.search(joint_q[s:e], K * 10)  # oversample, then mask
    for i in range(s, e):
        cand_ids, cand_sims = I[i - s], D[i - s]
        m = (meta[cand_ids] < dates_q[i])
        idx = np.where(m)[0]
        if idx.size == 0:
            no_past_cnt += 1
            sel = np.arange(min(K, len(cand_ids)))
        else:
            sel = idx[:K]
        ids = cand_ids[sel]
        z_retr[i] = text_train[ids].mean(axis=0)
        eff_k_hist.append(len(ids))

t1 = time.time()
print(f"Retrieval done in {t1 - t0:.2f}s | no-past fallbacks: {no_past_cnt}")

# ---------- Save OOD with z_retr (normalize to lists first) ----------
ood_out = ood.copy()
ood_out["text_embed"] = embeds.apply(lambda a: a.astype("float32").tolist())
ood_out["z_retr"]     = [vec_to_list(v) for v in z_retr]

ood_out.to_parquet(ood_out_path, index=False)
print("Saved:", ood_out_path)

# ---------- Causal numeric restandardization (leakage-safe, adds *_rollz) ----------
# Window = last 252 trading days from TRAIN strictly before OOD start
train_df = pd.read_parquet(train_feat, engine="pyarrow").sort_values("Date")
ood_start = pd.to_datetime(ood_out["Date"].min()).normalize()
ref = train_df[train_df["Date"] < ood_start]
ref_win = ref.tail(252)  # ~1Y business days

# Pick numeric-only columns for restandardization (exclude label/text/retrieval/macros/realized return)
drop_exact = {
    "Date","Movement","text_embed","z_retr",
    "cpi_yoy_lagged_z","unrate_lagged_z","t10y2y_lagged_z","gdp_qoq_lagged_z",
    "Daily_Return"
}
num_cols_roll = [c for c in ood_out.columns if c not in drop_exact and is_numeric_series(ood_out[c])]

mu = ref_win[num_cols_roll].mean(numeric_only=True)
sd = ref_win[num_cols_roll].std(ddof=1, numeric_only=True).replace(0.0, np.nan)

added = 0
for c in num_cols_roll:
    rollz = (ood_out[c] - mu.get(c, np.nan)) / sd.get(c, np.nan)
    ood_out[c + "_rollz"] = rollz.fillna(0.0).astype("float32")
    added += 1
print(f"[rollz] Added {added} *_rollz columns using {len(ref_win)}-row reference window ending {ref_win['Date'].max().date() if len(ref_win) else 'N/A'}")

# ---------- Build features parquet (keep raw + *_rollz + macros + embeds/retrieval) ----------
numeric_cols = [
    c for c in ood_out.columns
    if c not in {"text_embed","z_retr"} and is_numeric_series(ood_out[c])
]

features = ood_out[["Date"] + numeric_cols].copy()
features["text_embed"] = ood_out["text_embed"].apply(vec_to_list)
features["z_retr"]     = ood_out["z_retr"].apply(vec_to_list)

features.to_parquet(features_out_path, index=False)
print("Saved:", features_out_path)

# ---------- Quick checks ----------
print("\n=== XOM TEST ARTIFACTS ===")
print("xom_test_ood.parquet shape:", pd.read_parquet(ood_out_path, engine="pyarrow").shape)
print("XOM_features.parquet shape :", pd.read_parquet(features_out_path, engine="pyarrow").shape)
eff_k = np.array(eff_k_hist, dtype=int)
print("Effective K (min/median/max):", int(eff_k.min()), float(np.median(eff_k)), int(eff_k.max()))
print("Used GPU:", used_gpu)
print("Sample embed lens:", len(features['text_embed'].iloc[0]), len(features['z_retr'].iloc[0]))

# Helpful hint for evaluation: select *_rollz in your numeric-only baseline
rollz_example = [c for c in features.columns if c.endswith("_rollz")][:8]
print("Example *_rollz cols:", rollz_example)
