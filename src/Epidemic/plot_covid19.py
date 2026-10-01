import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = Path(__file__).stem

path = Path(__file__).with_name("COVID19_total_cases.csv")

df = pd.read_csv(path, encoding="utf-8-sig")

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df = df.sort_values("date").reset_index(drop=True)

def make_data(start=None, end=None):
    sub = df.copy()

    if start is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start)]

    if end is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end)]

    sub = sub.reset_index(drop=True)

    ts = jnp.array((sub["date"] - sub["date"].iloc[0]).dt.days.values)

    ys = jnp.array(sub["cases"].values)

    return ts, ys

if __name__ == "__main__":
    ts, ys = make_data("2022-01", "2023-04")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(ts, ys, label="data", s=4, color="tab:blue")
    ax.legend()
    ax.set_xlabel("days")
    ax.set_ylabel("incidence")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "COVID19_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

