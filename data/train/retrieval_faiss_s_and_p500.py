# ==== Retrieval: build z_retr for S&P500 TRAIN and save sp500_features_with_zretr.parquet ====
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Paths ----------
ROOT   = Path("/home/s6skkhan/macro_retrieval")
TRAIND = ROOT / "train"

index_path   = TRAIND / "index_flat_ip.faiss"
meta_path    = TRAIND / "index_meta.parquet"
train_feat   = TRAIND / "sp500_features.parquet"   # <-- TRAIN base (has Daily_Return)
text_mat     = TRAIND / "text_train.npy"                    # optional, faster if present
train_out    = TRAIND / "sp500_features_with_zretr.parquet" # <-- OUTPUT (this script creates)

alpha = 0.5    # macro weight in the joint vector (same α you used for the index)
K = 5          # neighbors to aggregate
B = 512        # batch size for FAISS search

# ---------- Load FAISS index (GPU if available) ----------
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

print(f"FAISS index loaded. ntotal={index.ntotal}, dim={index.d}, GPU={used_gpu}")

# ---------- Load meta (dates) ----------
meta = pd.read_parquet(meta_path, engine="pyarrow")["date"].to_numpy(dtype="datetime64[ns]")
print("Meta dates loaded:", meta.shape)

# ---------- Load TRAIN features ----------
train_df = pd.read_parquet(train_feat, engine="pyarrow").sort_values("Date").reset_index(drop=True)
print("Loaded TRAIN base:", train_df.shape, "Dates:", train_df["Date"].min().date(), "→", train_df["Date"].max().date())

# ---------- Load TRAIN text embeddings (neighbor pool for aggregation) ----------
if text_mat.exists():
    text_pool = np.load(text_mat).astype("float32")  # shape (N, 384)
    print("Loaded text_train.npy:", text_pool.shape)
else:
    text_pool = np.vstack(train_df["text_embed"].to_numpy()).astype("float32")
    print("Built text_pool from train_df['text_embed']:", text_pool.shape)

# ---------- Build normalized joint queries [text ; α*macro] for each TRAIN day ----------
macro_cols = ["cpi_yoy_lagged_z","unrate_lagged_z","t10y2y_lagged_z","gdp_qoq_lagged_z"]
text_q  = np.vstack(train_df["text_embed"].to_numpy()).astype("float32")
macro_q = train_df[macro_cols].to_numpy().astype("float32")
joint_q = np.concatenate([text_q, alpha * macro_q], axis=1).astype("float32")
joint_q /= (np.linalg.norm(joint_q, axis=1, keepdims=True) + 1e-9)

print("Train joint shape:", joint_q.shape, "(expect [N_train, 388])")

# ---------- FAISS batched search with causal masking (date_neighbor < date_query) ----------
t0 = time.time()
dates_q = train_df["Date"].to_numpy(dtype="datetime64[ns]")
z_retr = np.zeros_like(text_q, dtype="float32")

no_past_cnt = 0
eff_k_hist = []

for s in range(0, joint_q.shape[0], B):
    e = min(s + B, joint_q.shape[0])
    D, I = index.search(joint_q[s:e], K * 10)  # oversample, we'll mask to date<t

    for i in range(s, e):
        cand_ids, cand_sims = I[i - s], D[i - s]
        # Causal mask: only neighbors strictly before the query date
        m = (meta[cand_ids] < dates_q[i])
        idx = np.where(m)[0]
        if idx.size == 0:
            # Fallback: first few earliest dates
            no_past_cnt += 1
            sel = np.arange(min(K, len(cand_ids)))
        else:
            sel = idx[:K]

        ids = cand_ids[sel]
        z_retr[i] = text_pool[ids].mean(axis=0)
        eff_k_hist.append(len(ids))

t1 = time.time()
print(f"TRAIN retrieval done in {t1 - t0:.2f}s. Fallbacks (no past neighbors): {no_past_cnt}")

# ---------- Save TRAIN with z_retr ----------
train_out_df = train_df.copy()
train_out_df["z_retr"] = list(z_retr.astype("float32"))
train_out_df.to_parquet(train_out, index=False)
print("Saved TRAIN with z_retr ->", train_out)

# ---------- Checks to paste back ----------
print("\n=== TRAIN RETRIEVAL CHECKS ===")
print("Rows:", len(train_out_df))
print("Columns:", list(train_out_df.columns))
print("Any NaNs in z_retr? ", any(pd.Series(train_out_df["z_retr"]).apply(lambda v: np.isnan(v).any())))
print("z_retr dim (first row):", len(train_out_df["z_retr"].iloc[0]))

# Effective K distribution
eff_k = np.array(eff_k_hist, dtype=int)
print("Effective K stats → min/median/max:", eff_k.min(), np.median(eff_k), eff_k.max())

# Spot-check 3 random rows for causality
rng = np.random.default_rng(7)
sample_idx = rng.integers(0, len(train_out_df), size=min(3, len(train_out_df)))
print("Sample indices:", sample_idx.tolist())

violations = 0
for j in sample_idx:
    D, I = index.search(joint_q[j:j+1], K * 10)
    cand_ids, cand_sims = I[0], D[0]
    m = (meta[cand_ids] < dates_q[j])
    ids = cand_ids[np.where(m)[0][:K]]
    sims = cand_sims[np.where(m)[0][:K]]
    if ids.size == 0:
        ids, sims = cand_ids[:K], cand_sims[:K]  # fallback for printing
    bad = (meta[ids] >= dates_q[j]).sum()
    violations += int(bad)
    print(f"  idx={int(j)} | date={pd.Timestamp(dates_q[j]).date()} | effK={len(ids)} | sims(top3)={np.round(sims[:3],4).tolist()} | all_past={bad==0}")

print("Causal violations in spot-checks:", violations)

# Basic similarity stats on joint vectors (sanity)
D0, I0 = index.search(joint_q[0:1], 1)
print("Top-1 sim for first TRAIN query:", float(D0[0][0]))
print("Used GPU:", used_gpu)
