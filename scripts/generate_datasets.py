"""
Simulate domain name data for a two-part "digital monocentric city" study.

Part I (Quasi-experimental):
- "chat" cluster panel around ChatGPT launch (Nov 2022)
- Fuzzy jump in attention (A_t) at cutoff
- Distance-modulated exposure
- Domain FE + time FE
- Selection into observed sales (GoDaddy-like registry)

Part II (Observational):
- Multiple keyword clusters (lawyers, hotel, finance, crypto, etc.)
- Cross-sectional gradients in prices vs semantic distance
- Gradients correlate with keyword popularity (city-size analogue)

NOTE:
- This script INVENTS domain names FIRST.
- It creates a placeholder "true_distance" for simulation.

Outputs:
- domains_master.csv
- chat_panel_all.csv
- chat_sales_panel.csv
- multi_keyword_cross_section.csv
- domain_names.txt
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Helper lists for name generation
ADJECTIVES = [
    "best", "top", "pro", "smart", "quick", "easy", "free", "online",
    "global", "local", "prime", "trusted", "secure", "modern", "instant",
    "expert", "elite", "fast", "simple", "direct", "true", "my"
]

MODIFIERS = [
    "hub", "zone", "world", "now", "today", "plus", "central",
    "network", "group", "service", "portal", "market", "guide",
    "connect", "assist", "help", "solutions", "experts"
]

CITIES = ["nyc", "la", "miami", "chicago", "boston", "seattle", "austin", "sf"]
INDUSTRY_SUFFIX = ["app", "site", "live", "ai", "bot", "team", "pro"]


# Weighted toward .com to mimic reality
TLD_POOL = [".com", ".net", ".org", ".io", ".co", ".ai"]
TLD_WEIGHTS = [0.72, 0.08, 0.06, 0.06, 0.05, 0.03]


# Domain name generators
def pick_tld():
    return random.choices(TLD_POOL, weights=TLD_WEIGHTS, k=1)[0]


def maybe_add_digit(s, p=0.12):
    if random.random() < p:
        return s + str(random.choice([1, 2, 24, 360, 101]))
    return s


def maybe_add_hyphen(tokens, p=0.10):
    if len(tokens) <= 1:
        return tokens[0]
    if random.random() < p:
        return "-".join(tokens)
    return "".join(tokens)


def generate_variants(keyword: str, n: int):
    """
    Generate n invented second-level names for a keyword cluster.
    """
    out = set()

    patterns = [
        lambda k: [random.choice(ADJECTIVES), k],
        lambda k: [k, random.choice(ADJECTIVES)],
        lambda k: [random.choice(ADJECTIVES), k, random.choice(MODIFIERS)],
        lambda k: [k, random.choice(MODIFIERS)],
        lambda k: [k, random.choice(CITIES)],
        lambda k: [random.choice(CITIES), k],
        lambda k: [k, random.choice(INDUSTRY_SUFFIX)],
        lambda k: [random.choice(ADJECTIVES), k, random.choice(INDUSTRY_SUFFIX)],
    ]

    while len(out) < n:
        toks = random.choice(patterns)(keyword)
        base = maybe_add_hyphen(toks)
        base = maybe_add_digit(base)

        # length within reasonable length
        if 4 <= len(base) <= 25:
            out.add(base.lower())

    return list(out)


def generate_cluster_domains(keyword: str, n_variants: int, include_center=True):
    """
    Create full domains for a keyword cluster.
    Returns list of full domain strings (with TLD).
    """
    domains = []

    if include_center:
        domains.append(f"{keyword}.com")

    variants = generate_variants(keyword, n_variants)

    for v in variants:
        tld = pick_tld()
        domains.append(v + tld)

    # Deduplicate 
    domains = list(dict.fromkeys(domains))
    return domains


def parse_domain(domain: str):
    for tld in sorted(TLD_POOL, key=len, reverse=True):
        if domain.endswith(tld):
            sld = domain[:-len(tld)]
            return sld, tld
    parts = domain.split(".")
    return parts[0], "." + parts[-1]


def token_count_rough(sld: str):
    """
    Rough token count:
    - split on hyphen if present
    - otherwise treat as 1 token
    """
    if "-" in sld:
        return len([t for t in sld.split("-") if t])
    return 1


def domain_covariates(domains: list, keyword_center_map: dict):
    """
    Build domain-level covariates.
    keyword_center_map: maps a domain to its cluster keyword.
    """
    rows = []
    for d in domains:
        sld, tld = parse_domain(d)
        rows.append({
            "domain": d,
            "sld": sld,
            "tld": tld,
            "is_com": int(tld == ".com"),
            "length": len(sld),
            "has_hyphen": int("-" in sld),
            "has_digit": int(any(ch.isdigit() for ch in sld)),
            "token_count": token_count_rough(sld),
            "cluster_keyword": keyword_center_map.get(d, None),
        })

    df = pd.DataFrame(rows)

    df["exact_match"] = (df["sld"] == df["cluster_keyword"]).astype(int)

    age = np.random.lognormal(mean=1.2, sigma=0.6, size=len(df))
    df["age_years"] = np.clip(age, 0, 30)

    return df


# Placeholder distance
def make_latent_distance(df: pd.DataFrame):
    """
    Create a plausible distance correlated with lexical quality:
    This is ONLY for simulation now.
    Later replaced with QWEN3 embedding distances.
    Nuance for generating other variables.
    """
    n = len(df)
    # Base distance distribution with mass near 0
    base = np.random.beta(a=2.0, b=5.0, size=n)

    # Penalize longer names, non-.com, hyphens, digits
    length_penalty = 0.015 * (df["length"] - df["length"].mean())
    tld_penalty = 0.08 * (1 - df["is_com"])
    hyphen_penalty = 0.06 * df["has_hyphen"]
    digit_penalty = 0.05 * df["has_digit"]

    noise = np.random.normal(0, 0.04, size=n)

    dist = base + length_penalty + tld_penalty + hyphen_penalty + digit_penalty + noise
    dist = np.clip(dist, 0, 1)

    # Force center domains to have distance 0
    df = df.copy()
    df["true_distance"] = dist
    df.loc[df["exact_match"] == 1, "true_distance"] = 0.0

    # Placeholder column 
    df["embedding_distance"] = np.nan

    return df


# Popularity indices
def make_keyword_popularity(keywords):
    """
    Cross-sectional popularity (city-size analogue).
    Nuance for generating price gradients.
    """
    pop = {k: float(np.clip(np.random.normal(0.6, 0.18), 0.2, 1.0)) for k in keywords}
    # make "chat" relatively high to match story
    if "chat" in pop:
        pop["chat"] = max(pop["chat"], 0.75)
    return pop


def make_attention_series(dates, cutoff="2022-11-01"):
    """
    Fuzzy attention jump for the chat category.
    Returns a pandas Series A_t.
    """
    cutoff = pd.Timestamp(cutoff)
    A = []
    for dt in dates:
        pre = dt < cutoff
        mu = 0.35 if pre else 0.75  # jump in mean attention
        A.append(np.random.normal(mu, 0.08))
    A = np.clip(A, 0.05, 1.2)
    return pd.Series(A, index=dates, name="attention_index")


# Part II: Observational cross-sectional prices
def simulate_cross_section(df, keyword_pop, 
                           b0=1.0, b1=0.9,
                           com_premium=0.35, 
                           length_coef=-0.06,
                           age_coef=0.015,
                           hyphen_pen=-0.12,
                           digit_pen=-0.10,
                           heavy_tail_p=0.015):
    """
    Cross-sectional log prices across keyword clusters.

    Gradient steepens with keyword popularity:
        slope_k = -(b0 + b1 * S_k)
    """
    df = df.copy()

    # keyword FE proxy
    df["S_k"] = df["cluster_keyword"].map(keyword_pop).astype(float)

    # keyword-specific gradient
    df["slope_k"] = -(b0 + b1 * df["S_k"])

    # latent quality FE
    df["alpha_i"] = np.random.normal(0, 0.35, size=len(df))

    # base level by keyword (amenity-like intercept)
    df["gamma_k"] = 1.2 + 0.8 * df["S_k"] + np.random.normal(0, 0.08, size=len(df))

    d = df["true_distance"].values

    # idiosyncratic error + occasional heavy tail shocks
    eps = np.random.normal(0, 0.25, size=len(df))
    tail = np.random.binomial(1, heavy_tail_p, size=len(df)) * np.random.normal(1.2, 0.4, size=len(df))
    eps = eps + tail

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

    # Convert to price
    df["price_latent"] = np.exp(df["log_price_latent"])

    # Selection into observed "registry" sales
    # Higher latent value implies higher chance of sale
    logits = -2.0 + 0.7 * df["log_price_latent"]
    sale_prob = 1 / (1 + np.exp(-logits))
    df["sale_prob"] = sale_prob
    df["sold"] = (np.random.uniform(0, 1, len(df)) < sale_prob).astype(int)

    # Observed price only if sold 
    trans_noise = np.random.normal(0, 0.08, size=len(df))
    df["log_price_obs"] = np.where(df["sold"] == 1, df["log_price_latent"] + trans_noise, np.nan)
    df["price_obs"] = np.where(df["sold"] == 1, np.exp(df["log_price_obs"]), np.nan)

    return df


# Part I: Chat panel with fuzzy attention jump + gradient change
def simulate_chat_panel(chat_df, 
                        start="2021-01-01", end="2024-12-01",
                        cutoff="2022-11-01",
                        kappa=2.2,
                        base_grad=1.1,
                        post_grad_add=0.7,
                        com_premium=0.35,
                        length_coef=-0.06,
                        age_coef=0.015,
                        hyphen_pen=-0.12,
                        digit_pen=-0.10):
    """
    Panel of chat domains with:
    - fuzzy attention A_t jump at cutoff
    - exposure T_it = A_t * exp(-kappa * d_i)
    - post-shock steepening of distance gradient
    - domain FE + time FE
    - selection into observed sales
    """
    chat_df = chat_df.copy()

    dates = pd.date_range(start=start, end=end, freq="MS")
    A_t = make_attention_series(dates, cutoff=cutoff)

    # Domain FE
    alpha_i = pd.Series(np.random.normal(0, 0.35, size=len(chat_df)),
                        index=chat_df["domain"]).to_dict()

    # Time FE (market-wide cycles)
    lambda_t = pd.Series(np.random.normal(0, 0.05, size=len(dates)),
                         index=dates).to_dict()

    rows = []
    for _, row in chat_df.iterrows():
        domain = row["domain"]
        d_i = float(row["true_distance"])
        is_com = int(row["is_com"])
        length = int(row["length"])
        age = float(row["age_years"])
        has_h = int(row["has_hyphen"])
        has_d = int(row["has_digit"])

        for dt in dates:
            post = int(dt >= pd.Timestamp(cutoff))

            # Exposure intensity (distance-modulated attention)
            T_it = float(A_t.loc[dt] * np.exp(-kappa * d_i))

            # Distance gradient pre vs post
            # slope = -(base_grad + post * post_grad_add)
            slope = -(base_grad + post * post_grad_add)

            # Idiosyncratic error
            eps = np.random.normal(0, 0.22)

            log_price_latent = (
                1.35  # baseline
                + alpha_i[domain]
                + lambda_t[dt]
                + slope * d_i
                + 0.45 * T_it  # attention raises value
                + com_premium * is_com
                + length_coef * np.log1p(length)
                + age_coef * np.log1p(age)
                + hyphen_pen * has_h
                + digit_pen * has_d
                + eps
            )

            # Sale probability (only observe sold)
            logits = -2.2 + 0.75 * log_price_latent
            sale_prob = 1 / (1 + np.exp(-logits))
            sold = int(np.random.uniform() < sale_prob)

            # Observed transaction noise
            trans_noise = np.random.normal(0, 0.08) if sold else np.nan
            log_price_obs = log_price_latent + trans_noise if sold else np.nan

            rows.append({
                "domain": domain,
                "date": dt,
                "post": post,
                "attention_index": float(A_t.loc[dt]),
                "true_distance": d_i,
                "embedding_distance": np.nan,  # you will fill
                "T_it": T_it,
                "is_com": is_com,
                "length": length,
                "age_years": age,
                "has_hyphen": has_h,
                "has_digit": has_d,
                "exact_match": int(row["exact_match"]),
                "alpha_i": alpha_i[domain],
                "lambda_t": lambda_t[dt],
                "log_price_latent": log_price_latent,
                "sale_prob": sale_prob,
                "sold": sold,
                "log_price_obs": log_price_obs,
                "price_obs": np.exp(log_price_obs) if sold else np.nan
            })

    panel = pd.DataFrame(rows)
    return panel


# Main pipeline
def main(output_dir="datasets/",
         n_chat_variants=240,
         n_other_variants=180):

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Keywords / clusters
    keywords = ["chat", "lawyers", "hotel", "finance", "crypto", "travel"]
    centers = {k: f"{k}.com" for k in keywords}

    # 1) Invent domain names 
    all_domains = []
    keyword_center_map = {}

    for k in keywords:
        n_var = n_chat_variants if k == "chat" else n_other_variants
        cluster_domains = generate_cluster_domains(k, n_var, include_center=True)
        all_domains.extend(cluster_domains)

        # map domain -> cluster keyword
        for d in cluster_domains:
            keyword_center_map[d] = k

    # Deduplicate domain names
    all_domains = list(dict.fromkeys(all_domains))

    with open(out_path / "domain_names.txt", "w", encoding="utf-8") as f:
        for d in all_domains:
            f.write(d + "\n")

    # 2) Build covariates table
    master = domain_covariates(all_domains, keyword_center_map)

    # 3) Create latent distances (placeholder)
    master = make_latent_distance(master)

    # Save master file
    master.to_csv(out_path / "domains_master.csv", index=False)

    # 4) Keyword popularity 
    keyword_pop = make_keyword_popularity(keywords)

    # 5) Part II: Observational cross section
    #    We simulate prices and keep only sold observations as the "registry"
    cross_df = simulate_cross_section(master, keyword_pop)
    # Keep sold rows only to mimic registry
    cross_sold = cross_df[cross_df["sold"] == 1].copy()

    # Save cross-sectional outputs
    cross_df.to_csv(out_path / "multi_keyword_cross_section_all.csv", index=False)
    cross_sold.to_csv(out_path / "multi_keyword_cross_section.csv", index=False)

    # 6) Part I: Quasi-experimental chat panel
    chat_master = master[master["cluster_keyword"] == "chat"].copy()
    chat_panel = simulate_chat_panel(chat_master)

    chat_panel_all = chat_panel.copy()
    chat_panel_sold = chat_panel[chat_panel["sold"] == 1].copy()

    chat_panel_all.to_csv(out_path / "chat_panel_all.csv", index=False)
    chat_panel_sold.to_csv(out_path / "chat_sales_panel.csv", index=False)

    # 7) Save files
    print("Saved files to:", out_path.resolve())
    print("Total invented domains:", len(master))
    print("Chat domains:", len(chat_master))
    print("Cross-sectional sold obs:", len(cross_sold))
    print("Chat panel sold obs:", len(chat_panel_sold))
    print("Keyword popularity S_k:", keyword_pop)


if __name__ == "__main__":
    main()
