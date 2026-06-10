import jax.numpy as jnp

import matplotlib.pyplot as plt

def relative_error(pred, true):
    return jnp.linalg.norm(pred - true, 2) / jnp.linalg.norm(true, 2)

def plotting(ts_data, ys_data, ts_eval, ys_eval, ys_pred, eta_eval, eta_pred, loss_list, viz_data):
    rel_Tu = relative_error(ys_pred[:, 0], ys_eval[:, 0])
    rel_Ti = relative_error(ys_pred[:, 1], ys_eval[:, 1])
    rel_T = relative_error(ys_pred[:, 0] + ys_pred[:, 1], ys_eval[:,0]+ys_eval[:,1])
    rel_V = relative_error(ys_pred[:, 2], ys_eval[:, 2])
    rel_ee = relative_error(eta_pred, eta_eval)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    if viz_data:
        axs[0][2].plot(ts_data, ys_data[:,0], ".", label="Data Total T")
        axs[1][0].plot(ts_data, ys_data[:,1], ".", label="Data Total V")

    axs[0][0].plot(ts_eval, ys_eval[:, 0], label="True T_U")
    axs[0][0].plot(ts_eval, ys_pred[:, 0], "--", label="Pred T_U")
    axs[0][0].set_title(f"T_U (Rel. Error: {rel_Tu:.2e})")
    axs[0][0].legend()

    axs[0][1].plot(ts_eval, ys_eval[:, 1], label="True T_I")
    axs[0][1].plot(ts_eval, ys_pred[:, 1], "--", label="Pred T_I")
    axs[0][1].set_title(f"T_I (Rel. Error: {rel_Ti:.2e})")
    axs[0][1].legend()

    axs[0][2].plot(ts_eval, ys_eval[:, 0]+ys_eval[:, 1], label="True Total T")
    axs[0][2].plot(ts_eval, ys_pred[:, 0] + ys_pred[:, 1], "--", label="Pred Total T")
    axs[0][2].set_title(f"T (Rel. Error: {rel_T:.2e})")
    axs[0][2].legend()

    axs[1][0].plot(ts_eval, ys_eval[:, 2], label="True V")
    axs[1][0].plot(ts_eval, ys_pred[:, 2], "--", label="Pred V")
    axs[1][0].set_title(f"V (Rel. Error: {rel_V:.2e})")
    axs[1][0].set_yscale("log")
    axs[1][0].legend()

    axs[1][1].plot(ts_eval, eta_eval, label="True eta")
    axs[1][1].plot(ts_eval, eta_pred, label="Pred eta", linestyle="--")
    axs[1][1].set_title(f"eta (Rel. Error 1: {rel_ee:.2e})")
    axs[1][1].set_ylim(-1e-4, 1e-4)
    axs[1][1].legend()

    axs[1][2].plot(loss_list)
    axs[1][2].set_yscale("log")
    axs[1][2].set_title("Training Loss")