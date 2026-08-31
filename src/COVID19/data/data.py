import os
import pandas as pd
import jax.numpy as jnp

path = os.path.join(os.getcwd(), "data/COVID19_data.csv")

df = pd.read_csv(
    path,
    sep=r"\s+",
    header=0,
    names=["date", "confirmed", "deaths", "recovered", "active"],
)

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df = df.sort_values("date").reset_index(drop=True)

def make_data(start_date=None, end_date=None):
    sub = df.copy()

    if start_date is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end_date)]

    sub = sub.reset_index(drop=True)

    ts = jnp.array((sub["date"] - sub["date"].iloc[0]).dt.days.values)

    ys = jnp.array(sub["active"].values).T

    days = ts[-1] - ts[0]

    ts = ts / days

    ys = ys / 5e+7

    return ts, ys, days

if __name__ == "__main__":

    _, _, x = make_data("2020-01-20", "2021-04-22")

    print(x)
    # print(y)