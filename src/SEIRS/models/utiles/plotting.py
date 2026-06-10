import jax.numpy as jnp

import matplotlib.pyplot as plt

def relative_error(pred, true):
    return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

def plotting(ts_data, ys_data, ts_eval, ys_pred, beta_pred, sigma_pred, loss_list):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(ts_data, ys_data, ".", label="True I")
    axs[0].plot(ts_eval, ys_pred[:, 2], "--", label="Pred I")
    axs[0].legend()

    axs[1].plot(ts_eval, beta_pred, label="Pred beta", linestyle="--")
    axs[1].legend()

    axs[2].plot(ts_eval, sigma_pred, label="Pred sigma", linestyle="--")
    axs[2].legend()

    # axs[3].plot(loss_list)
    # axs[3].set_yscale("log")
    # axs[3].set_title("Training Loss")