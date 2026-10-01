import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = Path(__file__).stem

def make_data():
    df = pd.read_csv(BASE_DIR / "influenza_sydney_1919.csv")
    ys = jnp.asarray(df["Total_Cases"].to_numpy(), dtype=jnp.float32)
    dates = pd.to_datetime("1919 " + df["Month"] + " " + df["Date"].astype(str),
                           format="%Y %B %d")
    ts = jnp.asarray((dates - dates.iloc[0]).dt.days.to_numpy(), dtype=jnp.float32)
    return ts, ys

if __name__ == "__main__":
    ts, ys = make_data()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(ts, ys, label="data", s=10, color="tab:blue")
    ax.legend()
    ax.set_xlabel("days")
    ax.set_ylabel("incidence")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "influenza_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig)