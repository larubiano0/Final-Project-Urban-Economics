# Digital Monocentric Cities  
### Domain Name Prices, Semantic Distance, and the Economics of Online Location

This repository contains the replication package for **“Digital Monocentric Cities: Domain Name Prices, Semantic Distance, and the Economics of Online Location.”**  
The project studies whether domain-name markets exhibit a **digital analogue of the monocentric city model**: within a semantic category (e.g., “chat”), do domain prices decline with **semantic distance** from a category “center” (e.g., `chat.com`), and does a major **attention shock** steepen this digital bid–rent curve?

We operationalize “virtual location” using modern **text embeddings** and cosine similarity, and evaluate two empirical components:

1. **Quasi-experimental panel (ChatGPT shock)**  
   A monthly panel around the public release of ChatGPT (Nov 2022) to estimate whether the **price–distance gradient steepens** after the shock using a **difference-in-differences event-study** with domain and time fixed effects.

2. **Cross-sectional multi-keyword evidence**  
   A cross-section spanning multiple keyword clusters (e.g., lawyers, finance, hotel, crypto, travel) to test whether prices exhibit **monotone negative gradients** with respect to semantic distance from each category center, and whether **more popular categories** display steeper gradients.

---

## Repository Structure

```

└── larubiano0-final-project-urban-economics/
    ├── LICENSE
    ├── datasets/
    │   ├── chat_google_trends.csv
    │   ├── domain_names.txt
    │   ├── domains_master.csv
    │   ├── domains_master_with_qwen.csv
    │   ├── chat_panel_all.csv
    │   ├── chat_panel_all_with_qwen.csv
    │   ├── chat_panel_all_qwen_prices.csv
    │   ├── chat_sales_panel.csv
    │   ├── chat_sales_panel_with_qwen.csv
    │   ├── chat_sales_panel_qwen_prices.csv
    │   ├── multi_keyword_cross_section.csv
    │   ├── multi_keyword_cross_section_with_qwen.csv
    │   ├── multi_keyword_cross_section_qwen_prices.csv
    │   ├── multi_keyword_cross_section_all.csv
    │   └── multi_keyword_cross_section_all_qwen_prices.csv
    ├── scripts/
    │   ├── generate_datasets.py
    │   ├── regenerate_prices.py
    │   ├── results.R
    │   ├── figures.py
    │   ├── compute_embeddings.ipynb
    │   └── Stats.ipynb
    └── writeup/
        ├── _config.yml
        ├── econsocart.cfg
        ├── econsocart.cls
        ├── ecta_sample.tex
        └── tables/
            └── tab_part1_main.tex


````

---

## Data Overview

### Units of observation
- **Domain name** is the fundamental unit.
- Domains are grouped into semantic clusters indexed by a keyword \(k\).
- Cluster centers are canonical exact-match domains (e.g., `chat.com`, `lawyers.com`).

### Datasets
- **Panel (chat cluster):** monthly domain-level panel (Jan 2021–Dec 2024).
  - `chat_panel_all.csv`: full panel (including non-sales / latent values if included by the DGP).
  - `chat_sales_panel.csv`: observed transactions (sales) panel.
- **Cross-section (multiple clusters):**
  - `multi_keyword_cross_section.csv`: one observation per domain (sales observed only for transacted domains if the DGP includes selection).

---

## Semantic Distance (“Virtual Location”)

We compute semantic distance using embeddings and cosine similarity:

\[
d_{ik} = \frac{1 - s_{ik}}{0.6},
\quad s_{ik} = \cos(\text{emb}(domain_i), \text{emb}(center_k)).
\]

Implementation uses `sentence-transformers` with a high-capacity embedding model and computes distances for every domain relative to its own cluster center.

---

## Reproducibility: Quick Start

### 1) Environment setup

We recommend a clean Python environment (conda or venv).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -U pip
pip install pandas numpy sentence-transformers torch
````

> If you use GPU, install the correct CUDA-enabled `torch` build per PyTorch instructions.

### 2) Generate datasets (DGP)

Creates the master list of domains, the panel around ChatGPT, and the multi-keyword cross-section.

```bash
python src/generate_datasets.py
```

### 3) Regenerate prices / sale selection

Applies the pricing model and generates observed transactions based on the sale process.

```bash
python src/regenerate_prices.py
```

### 4) Compute embedding-based distances

Produces `domains_master_with_qwen.csv` and merges `embedding_distance` into panel/cross-section files.

```bash
python src/compute_embedding_distance_qwen.py
```

Outputs are saved to `output/` (or the working directory, depending on the script paths).

---

## Main Empirical Specifications

### Part I: ChatGPT shock (panel)

Baseline fixed-effects DiD:

[
\log P_{it} = \alpha_i + \lambda_t + \beta_1 d_i + \beta_2 Post_t + \beta_3 (Post_t \times d_i) + \varepsilon_{it}.
]

* (\alpha_i): domain fixed effects
* (\lambda_t): month fixed effects
* (d_i): embedding distance to `chat.com`
* (Post_t): indicator for months ≥ Nov 2022
* **Coefficient of interest:** (\beta_3) (steepening of the gradient)

Event-study variant interacts distance with month dummies around the shock.

### Part II: Multi-keyword cross-section

For each cluster (k):

[
\log P_{ik} = \gamma_k + \delta_k d_{ik} + X_{ik}'\psi + \xi_{ik}.
]

A negative (\delta_k) indicates a bid–rent-like gradient.

---

## Output and Figures

After running the pipeline, you should be able to reproduce:

* The **post-ChatGPT steepening** of the price–distance gradient in the `chat` cluster
* The **negative cross-sectional gradients** across keyword clusters
* Summary tables and diagnostic plots (e.g., distance distributions, pre-trends, event-study slopes)

---

## Notes on Practical Execution

* Embedding computation can be slow on CPU for large domain sets.
* If runtime is an issue, you can:

  * compute embeddings in batches,
  * cache embeddings to disk,
  * switch to GPU where available.

---

## Citation

If you use or build on this codebase, please cite the paper:

> Rubiano Guerrero, L. A., Castillo Cabrera, C. A., & Rosas Castillo, A. F. (2025).
> *Digital Monocentric Cities: Domain Name Prices, Semantic Distance, and the Economics of Online Location.* Working paper.

---

## License

This repository is intended for academic use. If you plan to reuse the code in a commercial setting, please contact the authors.

---

## Contact

* Luis Alejandro Rubiano Guerrero — [la.rubiano@uniandes.edu.co](mailto:la.rubiano@uniandes.edu.co)
* Carlos Andrés Castillo Cabrera — [ca.castilloc1@uniandes.edu.co](mailto:ca.castilloc1@uniandes.edu.co)
* Andrés Felipe Rosas Castillo — [a.rosasc@uniandes.edu.co](mailto:a.rosasc@uniandes.edu.co)


