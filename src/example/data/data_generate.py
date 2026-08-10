import diffrax
import matplotlib.pyplot as plt
from .func_list import *

def data_generate(exp):
    func, ts, y0, dim = exp()

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(func),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=50000,
    )

    return y0, sol.ts, sol.ys, dim


if __name__ == "__main__":
    _, t, y, _ = data_generate(exp3)

    plt.figure(figsize=(6, 4))
    plt.plot(t, y, label=r"$y(t)$")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()