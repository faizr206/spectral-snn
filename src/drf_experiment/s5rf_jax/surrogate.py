from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def heaviside(x: jax.Array) -> jax.Array:
    return (x > 0).astype(jnp.float32)


def _with_surrogate_gradient(fwd: Callable[[jax.Array], jax.Array], bwd: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    @jax.custom_gradient
    def wrapped(x: jax.Array) -> tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
        return fwd(x), lambda g: (g * bwd(x),)

    return wrapped


def spike_surrogate_multi_gaussian(h: float = 0.15, sigma: float = 0.5, s: float = 6.0) -> Callable[[jax.Array], jax.Array]:
    def grad_multi_gaussian(x: jax.Array) -> jax.Array:
        gaussian = lambda value, mu, std: jnp.exp(-0.5 * ((value - mu) / std) ** 2)
        return (1 + h) * gaussian(x, 0, sigma) - h * gaussian(x, sigma, s * sigma) - h * gaussian(x, -sigma, s * sigma)

    return _with_surrogate_gradient(heaviside, grad_multi_gaussian)


def cartesian_spike(x: jax.Array) -> jax.Array:
    return spike_surrogate_multi_gaussian()(x.real - 1.0)


def polar_spike(x: jax.Array) -> jax.Array:
    return spike_surrogate_multi_gaussian()(jnp.abs(x) - 1.0)
