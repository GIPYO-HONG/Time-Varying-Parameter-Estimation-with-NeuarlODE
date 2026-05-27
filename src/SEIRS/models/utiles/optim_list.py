import optax

def adam(lr):
    optimizer = optax.adam(lr)
    return optimizer

def adam_clipping(lr, clip=1.0):
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(lr)
    )
    return optimizer

def adamw(lr, wd=1e-5):
    optimizer = optax.adamw(
        learning_rate=lr,
        weight_decay=wd,
    )
    return optimizer