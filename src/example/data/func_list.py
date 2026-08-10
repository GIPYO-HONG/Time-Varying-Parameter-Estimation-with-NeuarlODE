import jax.numpy as jnp

def exp1():

    def func(t, y, args=None):
        return 8.0 * (1-y) * y

    ts = jnp.linspace(0, 1, 100)
    y0 = jnp.array([0.02])

    dim = 1

    return func, ts, y0, dim

def exp2():

    def func(t, y, args=None):
        return jnp.cos(4*jnp.pi*y)

    ts = jnp.linspace(0, 1, 100)
    y0 = jnp.array([0.01])

    dim = 1

    return func, ts, y0, dim

def exp3():

    def func(t, y, args=None):
        x, z = y
        aa, bb, gg, dd = 12.5, 12.5, 12.5, 12.5
        dx = aa * x - bb * x * z
        dz = dd * x * z - gg * z
        return jnp.array([dx, dz])

    ts = jnp.linspace(0., 1., 100)
    y0 = jnp.array([0.6, 0.3])

    dim = 2

    return func, ts, y0, dim