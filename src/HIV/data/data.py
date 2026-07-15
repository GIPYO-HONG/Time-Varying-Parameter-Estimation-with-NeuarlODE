#reference: https://sph.uth.edu/dept/bads/faculty-home/hulinwu/datasets/actg315longitudinaldataviralload
import os
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

# 파일 로드
path = os.path.join(os.getcwd(), "HIV/data/ACTG315.csv")
df = pd.read_csv(path, sep=r"\s+", header=0,
                 names=["obs_no", "patid", "day", "log10_rna", "cd4"])
df = df.astype({"patid": int, "day": float, "log10_rna": float, "cd4": float})
df = df.sort_values(["patid", "day"]).reset_index(drop=True)

# 시각화
def plot_patient(patid):
    sub = df[df["patid"] == patid]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Patient {patid}")

    axes[0].plot(sub["day"], sub["log10_rna"], marker="o", color="#1D9E75", linestyle="None",)
    axes[0].set_title("HIV RNA (log₁₀)")
    axes[0].set_xlabel("Day")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(sub["day"], sub["cd4"], marker="s", color="#378ADD", linestyle="None",)
    axes[1].set_title("CD4 T-cell count")
    axes[1].set_xlabel("Day")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def make_data(patid):
    sub = df[df["patid"] == patid]

    ts = jnp.array(sub["day"])

    T = jnp.array(sub["cd4"])
    V = 10**jnp.array(sub["log10_rna"])
    ys = jnp.stack([T, V], axis=1)

    y0 = ys[0,:]

    scale_ = jnp.max(ys, axis=0) + 1e-6
    scale = jnp.array([scale_[0], scale_[0], scale_[1]])
    return ts, ys, y0, scale

if __name__=="__main__":
    patid = 16 #1~45
    plot_patient(patid)