import jax.numpy as jnp
import equinox as eqx

def eval(run):
    ts_data = run.ts
    ys_data = run.ys

    ts_eval = jnp.linspace(ts_data[0], ts_data[-1], len(ts_data) * 4 - 1)

    EX = run.EX

    exp_name = run.exp_name
    path = f"results/{exp_name}"
    param_path = f"{path}/model_parameter/best_parameter.eqx"

    EX.model = eqx.tree_deserialise_leaves(
                param_path,
                EX.model
            )

    ys_pred, pp_pred = EX.eval(ts_eval)

    return ts_data, ys_data, ts_eval, ys_pred, pp_pred

if __name__=="__main__":
    from run import run3

    ts_data, ys_data, ts_eval, ys_pred, pp_pred = eval(run3)

    print(pp_pred.shape)

