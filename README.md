# ⏳ History Rhymes: Macro-Contextual Retrieval for Robust Financial Forecasting

-----

## 🧩 Overview

**History Rhymes** introduces a **macro-contextual retrieval framework** for financial forecasting that grounds each prediction in **historically analogous macroeconomic regimes**.
By unifying **financial news sentiment** and **structured macro indicators** in a shared embedding space, the model retrieves precedent periods that mirror the current market state, enabling **robust, interpretable, and profitable forecasts under regime shift**.

> “Financial history may not repeat, but it often *rhymes*.”
> This project operationalizes that idea through retrieval-augmented reasoning.

-----

## 🚀 Key Contributions

  * **Macro-Aware Retrieval:**
    Jointly embeds daily financial news and macro indicators (CPI, unemployment, yield spread, GDP) for causal similarity search using FAISS.
  * **Retrieval-Augmented Forecasting:**
    Each prediction is conditioned on its most similar historical regimes instead of relying solely on static model parameters.
  * **Robustness Under Distribution Shift:**
    Trained on **S\&P 500 (2007-2023)** and evaluated OOD on **AAPL (2024)** and **XOM (2024)**, achieving:
      * AAPL 2024 → Profit Factor = **1.18**, Sharpe = **0.95**
      * XOM 2024 → Profit Factor = **1.16**, Sharpe = **0.61**
      * Narrowest CV→OOD gap ($\Delta$F1, $\Delta$Sharpe) among all tested baselines.
  * **Interpretability by Analogy:**
    Retrieved neighbors align with recognizable macro contexts (e.g., inflationary phases, yield-curve inversions), creating transparent evidence chains.

-----

## 🧠 Methodology

| Component | Description |
| :--- | :--- |
| **Text Encoder** | MiniLM fine-tuned on Financial PhraseBank + FiQA sentiment datasets |
| **Macro Variables** | CPI YoY, Unemployment Rate, 10Y-2Y Yield Spread, GDP QoQ |
| **Retriever** | FAISS (cosine similarity, causal masking) |
| **Forecaster** | Logistic Regression on \`\[numeric features ; retrieved context]\` |
| **Evaluation** | CV (2007-23) + OOD (AAPL/XOM 2024); metrics: F1, MCC, AUROC, Sharpe, Profit Factor |


-----

## 🏗️ Repository Structure

```
history_rhymes/
│
├── data/                # Processed datasets (S&P500, AAPL, XOM)
├── code/                # Core Python scripts (retrieval, training, evaluation)
├── results/             # Diagrams and result visualizations
└── README.md
```

-----

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/sarthak-12/history_rhymes.git
cd history_rhymes
```

Outputs include:

  * Retrieved historical analogues
  * Evaluation metrics
  * Diagnostic plots 

-----

## 🧾 Citation

If you use this work, please use the appropriate reference.


## 📬 Contact

Sarthak Khanna

University of Bonn · Fraunhofer IAIS
