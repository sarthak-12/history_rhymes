# ==== Retrieval: build z_retr for AAPL OOD and save x_test_ood.parquet ====
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
TESTD  = ROOT / "test"

index_path = TRAIND / "index_flat_ip.faiss"
meta_path  = TRAIND / "index_meta.parquet"
train_feat = TRAIND / "sp500_features.parquet"
text_mat   = TRAIND / "text_train.npy"                # optional, but faster if present
train_jv   = TRAIND / "train_joint_vectors.parquet"   # for quick integrity checks

ood_base_path = TESTD / "x_test_ood_base.parquet"
ood_out_path  = TESTD / "x_test_ood.parquet"

alpha = 0.5    # macro weight in the joint vector
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

# ---------- Load train text embeddings (for aggregating neighbors) ----------
if text_mat.exists():
    text_train = np.load(text_mat).astype("float32")
    print("Loaded text_train.npy:", text_train.shape)
else:
    # fallback: read from sp500_features.parquet
    tr = pd.read_parquet(train_feat, engine="pyarrow")
    text_train = np.vstack(tr["text_embed"].to_numpy()).astype("float32")
    print("Built text_train from sp500_features.parquet:", text_train.shape)

# ---------- Load OOD base (pre-retrieval) ----------
ood = pd.read_parquet(ood_base_path, engine="pyarrow").sort_values("Date").reset_index(drop=True)
print("Loaded OOD base:", ood.shape, "Dates:", ood["Date"].min().date(), "→", ood["Date"].max().date())

macro_cols = ["cpi_yoy_lagged_z","unrate_lagged_z","t10y2y_lagged_z","gdp_qoq_lagged_z"]

# ---------- Build normalized joint queries [text ; α*macro] ----------
text_q  = np.vstack(ood["text_embed"].to_numpy()).astype("float32")
macro_q = ood[macro_cols].to_numpy().astype("float32")
joint_q = np.concatenate([text_q, alpha * macro_q], axis=1).astype("float32")
joint_q /= (np.linalg.norm(joint_q, axis=1, keepdims=True) + 1e-9)

print("Query joint shape:", joint_q.shape, "(expect [T, 388])")

# ---------- FAISS batched search with causal masking ----------
t0 = time.time()
dates_q = ood["Date"].to_numpy(dtype="datetime64[ns]")
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
            # Fallback: no past neighbors (should be rare for earliest dates)
            no_past_cnt += 1
            sel = np.arange(min(K, len(cand_ids)))
        else:
            sel = idx[:K]

        ids = cand_ids[sel]
        z_retr[i] = text_train[ids].mean(axis=0)
        eff_k_hist.append(len(ids))

t1 = time.time()
print(f"Retrieval done in {t1 - t0:.2f}s. Fallbacks (no past neighbors): {no_past_cnt}")

# ---------- Save OOD with z_retr ----------
ood_out = ood.copy()
ood_out["z_retr"] = list(z_retr.astype("float32"))
ood_out.to_parquet(ood_out_path, index=False)
print("Saved OOD with z_retr ->", ood_out_path)

# ---------- Checks to paste back ----------
print("\n=== RETRIEVAL CHECKS ===")
print("Rows:", len(ood_out))
print("Columns:", list(ood_out.columns))
print("Any NaNs in z_retr? ", any(pd.Series(ood_out["z_retr"]).apply(lambda v: np.isnan(v).any())))
print("z_retr dim (first row):", len(ood_out["z_retr"].iloc[0]))

# Effective K distribution
eff_k = np.array(eff_k_hist, dtype=int)
print("Effective K stats → min/median/max:", eff_k.min(), np.median(eff_k), eff_k.max())

# Spot-check 3 random rows: ensure neighbors are from the past (uses the same search once more)
rng = np.random.default_rng(7)
sample_idx = rng.integers(0, len(ood_out), size=min(3, len(ood_out)))
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
    # Check causal condition
    bad = (meta[ids] >= dates_q[j]).sum()
    violations += int(bad)
    print(f"  idx={int(j)} | date={pd.Timestamp(dates_q[j]).date()} | effK={len(ids)} | sims(top3)={np.round(sims[:3],4).tolist()} | all_past={bad==0}")

print("Causal violations in spot-checks:", violations)

# Basic similarity stats on joint vectors (sanity)
# (recompute top1 sim for first row)
D0, I0 = index.search(joint_q[0:1], 1)
print("Top-1 sim for first query:", float(D0[0][0]))
print("Used GPU:", used_gpu)
