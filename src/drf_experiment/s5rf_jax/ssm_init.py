from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh


def make_hippo(n: int) -> jax.Array:
    p = jnp.sqrt(1 + 2 * jnp.arange(n))
    a = p[:, jnp.newaxis] * p[jnp.newaxis, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(n))
    return -a


def make_nplr_hippo(n: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    hippo = make_hippo(n)
    p = jnp.sqrt(jnp.arange(n) + 0.5)
    b = jnp.sqrt(2 * jnp.arange(n) + 1.0)
    return hippo, p, b


def make_dplr_hippo(n: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    a, p, b = make_nplr_hippo(n)
    s = a + p[:, jnp.newaxis] * p[jnp.newaxis, :]
    lambda_real = jnp.mean(jnp.diagonal(s)) * jnp.ones_like(jnp.diagonal(s))
    lambda_imag, v = eigh(s * -1j)
    p = v.conj().T @ p
    b_orig = b
    b = v.conj().T @ b
    return lambda_real + 1j * lambda_imag, p, b, v, b_orig


def init_a(block_size: int, num_blocks: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    lam, _, _, v, _ = make_dplr_hippo(block_size)
    vinv = v.conj().T
    lam = (lam * jnp.ones((num_blocks, block_size))).ravel()
    v = jax.scipy.linalg.block_diag(*([v] * num_blocks))
    vinv = jax.scipy.linalg.block_diag(*([vinv] * num_blocks))
    return jnp.stack([lam.real, lam.imag], axis=-1), v, vinv


def log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1):
    def init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return random.uniform(key, shape) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)

    return init


def init_log_steps(key: jax.Array, shape_and_range: tuple[int, float, float]) -> jax.Array:
    hidden, dt_min, dt_max = shape_and_range
    steps = []
    for _ in range(hidden):
        key, subkey = random.split(key)
        steps.append(log_step_initializer(dt_min, dt_max)(subkey, shape=(1,)))
    return jnp.array(steps)


def real_to_complex(x: jax.Array) -> jax.Array:
    return x[..., 0] + 1j * x[..., 1]


def complex_to_real(x: jax.Array) -> jax.Array:
    return jnp.stack([x.real, x.imag], axis=-1)


def trunc_standard_normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    out_dim, in_dim, _ = shape
    values = []
    for _ in range(out_dim):
        key, subkey = random.split(key)
        values.append(lecun_normal()(subkey, shape=(1, in_dim, 2)))
    return jnp.array(values)[:, 0]


def init_dense_vinvb(key: jax.Array, in_dim: int, out_dim: int, vinv: jax.Array) -> jax.Array:
    if out_dim != vinv.shape[1]:
        raise ValueError(f"out_dim={out_dim} must match Vinv width={vinv.shape[1]}")
    weights = real_to_complex(trunc_standard_normal(key, (out_dim, in_dim, 2)))
    return complex_to_real(vinv @ weights)
