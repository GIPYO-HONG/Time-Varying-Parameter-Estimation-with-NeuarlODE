from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

path = BASE_DIR / "ACTG315.csv"
df = pd.read_csv(path, sep=r"\s+", header=0,
                 names=["obs_no", "patid", "day", "log10_rna", "cd4"])
df = df.astype({"patid": int, "day": float, "log10_rna": float, "cd4": float})
df = df.sort_values(["patid", "day"]).reset_index(drop=True)

def plot(patid):
    sub = df[df["patid"] == patid]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Patient {patid}")

    axes[0].scatter(sub["day"], sub["log10_rna"], s=10, color="tab:blue")
    axes[0].set_title("HIV RNA (log₁₀)")
    axes[0].set_xlabel("Day")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].scatter(sub["day"], sub["cd4"], s=10, color="tab:green")
    axes[1].set_title("CD4 T-cell count")
    axes[1].set_xlabel("Day")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(BASE_DIR / f"data_patient_{patid}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    patid = 10
    plot(patid)