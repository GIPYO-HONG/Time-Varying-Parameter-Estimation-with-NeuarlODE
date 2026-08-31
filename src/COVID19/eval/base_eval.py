import jax.numpy as jnp
import equinox as eqx
import numpy as np

def eval(run):
    ts_data = run.ts
    ts_eval = jnp.linspace(ts_data[0], ts_data[-1], len(ts_data) * 4 - 1)

    EX = run.EX

    exp_name = run.exp_name
    path = f"results/{exp_name}"
    param_path = f"{path}/model_parameter/best_parameter.eqx"
    loss_list = np.load(f"{path}/loss_list.npy")

    EX.model = eqx.tree_deserialise_leaves(
                param_path,
                EX.model
            )

    fig = EX.plotting(ts_eval, loss_list)

    return fig

if __name__=="__main__":
    from run import run as rn

    EX = rn.EX

    exp_name = rn.exp_name
    path = f"results/{exp_name}"
    # param_path = f"{path}/model_parameter/best_parameter.eqx"
    # loss_list = np.load(f"{path}/loss_list.npy")
    param_path = f"{path}/model_parameter/model_step_77000.eqx"
    loss_list = []

    EX.model = eqx.tree_deserialise_leaves(
                param_path,
                EX.model
            )

    print(EX.model.y0)
