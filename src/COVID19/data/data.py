import os
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

path = os.path.join(os.getcwd(), "data/COVID19_total_cases.csv")

df = pd.read_csv(
    path,
    sep=r"\s+",
    header=0,
    names=["date", "cases"],
)

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df = df.sort_values("date").reset_index(drop=True)

def plot():
    plt.figure(figsize=(12, 4))

    plt.plot(
        df["date"],
        df["cases"],
        color="#1D9E75",
        linewidth=1.5,
    )

    plt.scatter(
        df["date"],
        df["cases"],
        color="#1D9E75",
        s=10,
    )

    plt.title("Daily COVID-19 Cases")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig("data/plot.png", dpi=300, bbox_inches="tight")

def make_data(start_date=None, end_date=None):
    sub = df.copy()

    if start_date is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end_date)]

    sub = sub.reset_index(drop=True)

    ts = jnp.array(
        (sub["date"] - sub["date"].iloc[0]).dt.days.values
    )

    ys = jnp.array(sub["cases"].values)[:, None].T

    days = ts[-1] - ts[0]

    ts = ts / days

    ys = ys / 5e+7

    return ts, ys, days

if __name__ == "__main__":

    plot()

    # ts, ys = make_data("2020-02-01", "2020-02-29")