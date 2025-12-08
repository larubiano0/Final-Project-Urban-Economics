import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
df = pd.read_csv("datasets/chat google trends.csv", skiprows=2)

date_col = df.columns[0]
value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
df = df.dropna(subset=[date_col, value_col])

# --- Plot style ---
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(14, 6))

# --- Line plot ---
ax.plot(
    df[date_col],
    df[value_col],
    linewidth=2,
    color="#1f77b4",
)


# --- Vertical line for ChatGPT release ---
release_date = pd.to_datetime("2022-11-30")

ax.axvline(
    release_date,
    color="red",
    linestyle="--",
    linewidth=2,
    alpha=0.8
)

ax.annotate(
    "ChatGPT-3.5\nPublic Release\n(Nov 2022)",
    xy=(release_date, df[value_col].max()*0.65),
    xytext=(release_date + pd.Timedelta(days=80),
            df[value_col].max()*0.85),
    arrowprops=dict(
        arrowstyle="->",
        color="red",
        lw=1.8
    ),
    fontsize=12,
    color="red",
    ha="left",
    va="center"
)

# --- Title and axis labels ---
ax.set_title(
    "Global Google Search Interest Over Time: “chat”",
    fontsize=18,
    pad=20
)
ax.set_ylabel("Search Interest Index (0–100)", fontsize=14)
ax.set_xlabel("Date", fontsize=14)

# --- X-axis formatting ---
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax.grid(alpha=0.25)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("figures/google_trends_chat.png", dpi=300)