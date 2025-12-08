"""
Regenerate simulated prices using embedding_distance (Qwen3-Embedding-8B)

What this script does:
1) Loads domains_master_with_qwen.csv.
2) Uses embedding_distance as the distance measure.
3) Regenerates:
   - Part II observational cross-section prices and sales registry
   - Part I chat quasi-experimental panel prices and sales registry
4) Tries to REUSE previously generated keyword popularity S_k, domain FE, time FE,
   and attention series.

Outputs:
- multi_keyword_cross_section_all_qwen_prices.csv
- multi_keyword_cross_section_qwen_prices.csv
- chat_panel_all_qwen_prices.csv
- chat_sales_panel_qwen_prices.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# Global defaults 
DEFAULTS = {
    # Cross-sectional
    "b0": 1.0,
    "b1": 0.9,
    "com_premium": 0.35,
    "length_coef": -0.06,
    "age_coef": 0.015,
    "hyphen_pen": -0.12,
    "digit_pen": -0.10,
    "heavy_tail_p": 0.015,

    # Chat panel
    "start": "2021-01-01",
    "end": "2024-12-01",
    "cutoff": "2022-11-01",
    "kappa": 2.2,
    "base_grad": 1.1,
    "post_grad_add": 0.7,
    "attention_jump_pre": 0.35,
    "attention_jump_post": 0.75,
    "attention_sd": 0.08,

    # Sale selection
    "xsec_sale_intercept": -2.0,
    "xsec_sale_slope": 0.7,
    "chat_sale_intercept": -2.2,
    "chat_sale_slope": 0.75,

    # Noise
    "xsec_eps_sd": 0.25,
    "xsec_trans_sd": 0.08,
    "chat_eps_sd": 0.22,
    "chat_trans_sd": 0.08,
}


# Utility 
def load_master(data_dir: Path):
    """
    Load master domains table with embedding_distance.
    Preference order:
      1) domains_master_with_qwen.csv
      2) domains_master.csv
    """
    p1 = data_dir / "domains_master_with_qwen.csv"
    p2 = data_dir / "domains_master.csv"

    if p1.exists():
        df = pd.read_csv(p1)
    elif p2.exists():
        df = pd.read_csv(p2)

    # Ensure required columns
    for col in ["domain", "cluster_keyword", "is_com", "length", "has_hyphen", "has_digit", "age_years", "exact_match"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in master: {col}")

    # Make a safe distance column for regeneration
    if "embedding_distance" not in df.columns:
        df["embedding_distance"] = np.nan
    if "true_distance" not in df.columns:
        df["true_distance"] = np.nan

    df["distance_used"] = df["embedding_distance"]
    # fallback to true_distance if embedding distance missing
    df.loc[df["distance_used"].isna(), "distance_used"] = df.loc[df["distance_used"].isna(), "true_distance"]

    # ensure exact matches are 0
    df.loc[df["exact_match"] == 1, "distance_used"] = 0.0

    return df


def infer_keyword_popularity_from_old(data_dir: Path):
    """
    Try to recover S_k used previously from the old cross-section file.
    """
    old_all = data_dir / "multi_keyword_cross_section_all.csv"
    if old_all.exists():
        df_old = pd.read_csv(old_all)
        if "cluster_keyword" in df_old.columns and "S_k" in df_old.columns:
            mapping = (
                df_old.groupby("cluster_keyword")["S_k"]
                .mean()
                .to_dict()
            )
            return mapping

def generate_keyword_popularity(keywords, seed=42):
    """
    Generate popularity mapping similar to prior simulator.
    """
    rng = np.random.default_rng(seed)
    pop = {}
    for k in keywords:
        val = float(np.clip(rng.normal(0.6, 0.18), 0.2, 1.0))
        pop[k] = val
    if "chat" in pop:
        pop["chat"] = max(pop["chat"], 0.75)
    return pop


def infer_alpha_i_from_old_chat(data_dir: Path):
    """
    Recover domain FE alpha_i from old chat panel.
    """
    old_chat = data_dir / "chat_panel_all.csv"
    if old_chat.exists():
        df_old = pd.read_csv(old_chat)
        if "domain" in df_old.columns and "alpha_i" in df_old.columns:
            mapping = (
                df_old.groupby("domain")["alpha_i"]
                .mean()
                .to_dict()
            )
            return mapping


def infer_lambda_t_from_old_chat(data_dir: Path):
    """
    Recover time FE lambda_t from old chat panel.
    """
    old_chat = data_dir / "chat_panel_all.csv"
    if old_chat.exists():
        df_old = pd.read_csv(old_chat)
        if "date" in df_old.columns and "lambda_t" in df_old.columns:
            df_old["date"] = pd.to_datetime(df_old["date"])
            mapping = (
                df_old.groupby("date")["lambda_t"]
                .mean()
                .to_dict()
            )
            return mapping

def infer_attention_from_old_chat(data_dir: Path):
    """
    Recover attention_index series from old chat panel.
    """
    old_chat = data_dir / "chat_panel_all.csv"
    if old_chat.exists():
        df_old = pd.read_csv(old_chat)
        if "date" in df_old.columns and "attention_index" in df_old.columns:
            df_old["date"] = pd.to_datetime(df_old["date"])
            series = (
                df_old.groupby("date")["attention_index"]
                .mean()
                .sort_index()
            )
            return series


def make_attention_series(dates, cutoff, pre_mu, post_mu, sd, rng):
    cutoff = pd.Timestamp(cutoff)
    A = []
    for dt in dates:
        mu = pre_mu if dt < cutoff else post_mu
        A.append(rng.normal(mu, sd))
    A = np.clip(A, 0.05, 1.2)
    return pd.Series(A, index=dates, name="attention_index")


# Regeneration
def regenerate_cross_section(master, keyword_pop, params, seed=42):
    rng = np.random.default_rng(seed)
    df = master.copy()

    # Popularity
    df["S_k"] = df["cluster_keyword"].map(keyword_pop).astype(float)

    # Keyword-specific gradient slope
    b0, b1 = params["b0"], params["b1"]
    df["slope_k"] = -(b0 + b1 * df["S_k"])

    # Domain FE alpha_i
    df["alpha_i"] = rng.normal(0, 0.35, size=len(df))

    # Keyword intercept (amenity-like)
    df["gamma_k"] = 1.2 + 0.8 * df["S_k"] + rng.normal(0, 0.08, size=len(df))

    d = df["distance_used"].values

    # Idiosyncratic error + heavy tail
    eps = rng.normal(0, params["xsec_eps_sd"], size=len(df))
    tail_flag = rng.binomial(1, params["heavy_tail_p"], size=len(df))
    tail = tail_flag * rng.normal(1.2, 0.4, size=len(df))
    eps = eps + tail

    # Hedonic controls
    com_premium = params["com_premium"]
    length_coef = params["length_coef"]
    age_coef = params["age_coef"]
    hyphen_pen = params["hyphen_pen"]
    digit_pen = params["digit_pen"]

    df["log_price_latent"] = (
        df["gamma_k"]
        + df["alpha_i"]
        + df["slope_k"] * d
        + com_premium * df["is_com"]
        + length_coef * np.log1p(df["length"])
        + age_coef * np.log1p(df["age_years"])
        + hyphen_pen * df["has_hyphen"]
        + digit_pen * df["has_digit"]
        + eps
    )

    # Selection into sales registry
    logits = params["xsec_sale_intercept"] + params["xsec_sale_slope"] * df["log_price_latent"]
    sale_prob = 1 / (1 + np.exp(-logits))
    df["sale_prob"] = sale_prob
    df["sold"] = (rng.uniform(0, 1, len(df)) < sale_prob).astype(int)

    # Observed transaction prices only if sold
    trans_noise = rng.normal(0, params["xsec_trans_sd"], size=len(df))
    df["log_price_obs"] = np.where(df["sold"] == 1, df["log_price_latent"] + trans_noise, np.nan)
    df["price_obs"] = np.where(df["sold"] == 1, np.exp(df["log_price_obs"]), np.nan)

    return df


# Regeneration
def regenerate_chat_panel(master, params, data_dir: Path, seed=42):
    rng = np.random.default_rng(seed)

    chat_master = master[master["cluster_keyword"] == "chat"].copy()
    if chat_master.empty:
        raise ValueError("No chat domains found in master.")

    # Dates
    dates = pd.date_range(start=params["start"], end=params["end"], freq="MS")

    # Reuse attention from old file
    attention_old = infer_attention_from_old_chat(data_dir)
    if attention_old is not None:
        # align to our date index; fill missing by forward/backward fill
        A_t = attention_old.reindex(dates).ffill().bfill()
    else:
        A_t = make_attention_series(
            dates=dates,
            cutoff=params["cutoff"],
            pre_mu=params["attention_jump_pre"],
            post_mu=params["attention_jump_post"],
            sd=params["attention_sd"],
            rng=rng
        )

    # Reuse domain FE alpha_i and time FE lambda_t
    alpha_map = infer_alpha_i_from_old_chat(data_dir)
    lambda_map = infer_lambda_t_from_old_chat(data_dir)

    if alpha_map is None:
        alpha_map = {d: float(rng.normal(0, 0.35)) for d in chat_master["domain"].tolist()}

    if lambda_map is None:
        lambda_map = {dt: float(rng.normal(0, 0.05)) for dt in dates}

    # Parameters
    cutoff = pd.Timestamp(params["cutoff"])
    kappa = params["kappa"]
    base_grad = params["base_grad"]
    post_grad_add = params["post_grad_add"]

    com_premium = params["com_premium"]
    length_coef = params["length_coef"]
    age_coef = params["age_coef"]
    hyphen_pen = params["hyphen_pen"]
    digit_pen = params["digit_pen"]

    rows = []
    for _, row in chat_master.iterrows():
        domain = row["domain"]
        d_i = float(row["distance_used"])
        is_com = int(row["is_com"])
        length = int(row["length"])
        age = float(row["age_years"])
        has_h = int(row["has_hyphen"])
        has_d = int(row["has_digit"])
        exact = int(row["exact_match"])

        for dt in dates:
            post = int(dt >= cutoff)

            # Exposure with new distance
            T_it = float(A_t.loc[dt] * np.exp(-kappa * d_i))

            # Steeper gradient post shock
            slope = -(base_grad + post * post_grad_add)

            eps = rng.normal(0, params["chat_eps_sd"])

            log_price_latent = (
                1.35
                + alpha_map.get(domain, 0.0)
                + lambda_map.get(dt, 0.0)
                + slope * d_i
                + 0.45 * T_it
                + com_premium * is_com
                + length_coef * np.log1p(length)
                + age_coef * np.log1p(age)
                + hyphen_pen * has_h
                + digit_pen * has_d
                + eps
            )

            # Selection into registry sales
            logits = params["chat_sale_intercept"] + params["chat_sale_slope"] * log_price_latent
            sale_prob = 1 / (1 + np.exp(-logits))
            sold = int(rng.uniform() < sale_prob)

            trans_noise = rng.normal(0, params["chat_trans_sd"]) if sold else np.nan
            log_price_obs = log_price_latent + trans_noise if sold else np.nan

            rows.append({
                "domain": domain,
                "date": dt,
                "post": post,
                "attention_index": float(A_t.loc[dt]),
                "embedding_distance": float(row.get("embedding_distance", np.nan)),
                "true_distance": float(row.get("true_distance", np.nan)),
                "distance_used": d_i,
                "T_it": T_it,
                "is_com": is_com,
                "length": length,
                "age_years": age,
                "has_hyphen": has_h,
                "has_digit": has_d,
                "exact_match": exact,
                "alpha_i": float(alpha_map.get(domain, 0.0)),
                "lambda_t": float(lambda_map.get(dt, 0.0)),
                "log_price_latent": log_price_latent,
                "sale_prob": float(sale_prob),
                "sold": sold,
                "log_price_obs": log_price_obs,
                "price_obs": float(np.exp(log_price_obs)) if sold else np.nan
            })

    panel = pd.DataFrame(rows)
    return panel


def main():
    parser = argparse.ArgumentParser(description="Regenerate prices using embedding_distance.")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Folder with simulation outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for regeneration.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.seed)

    # Load master
    master = load_master(data_dir)

    # Keywords
    keywords = sorted(master["cluster_keyword"].dropna().unique().tolist())

    # Reuse popularity mapping
    keyword_pop = infer_keyword_popularity_from_old(data_dir)
    if keyword_pop is None:
        keyword_pop = generate_keyword_popularity(keywords, seed=args.seed)

    # Cross-section regeneration 
    xsec_all = regenerate_cross_section(master, keyword_pop, DEFAULTS, seed=args.seed)
    xsec_sold = xsec_all[xsec_all["sold"] == 1].copy()

    xsec_all.to_csv(data_dir / "multi_keyword_cross_section_all_qwen_prices.csv", index=False)
    xsec_sold.to_csv(data_dir / "multi_keyword_cross_section_qwen_prices.csv", index=False)

    # Chat panel regeneration
    chat_panel_all = regenerate_chat_panel(master, DEFAULTS, data_dir=data_dir, seed=args.seed)
    chat_panel_sold = chat_panel_all[chat_panel_all["sold"] == 1].copy()

    chat_panel_all.to_csv(data_dir / "chat_panel_all_qwen_prices.csv", index=False)
    chat_panel_sold.to_csv(data_dir / "chat_sales_panel_qwen_prices.csv", index=False)

    print("Regeneration complete.")
    print("Saved:")
    print(" -", data_dir / "multi_keyword_cross_section_all_qwen_prices.csv")
    print(" -", data_dir / "multi_keyword_cross_section_qwen_prices.csv")
    print(" -", data_dir / "chat_panel_all_qwen_prices.csv")
    print(" -", data_dir / "chat_sales_panel_qwen_prices.csv")
    print()
    print("Counts:")
    print("Total domains:", len(master))
    print("Cross-section sold:", len(xsec_sold))
    print("Chat panel sold:", len(chat_panel_sold))
    print("Keywords:", keywords)


if __name__ == "__main__":
    main()
