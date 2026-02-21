# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Tuple

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Discrete:
    """
    Discrete action space — `n` equally-likely integer actions.

    Parameters
    ----------
    n : int
        Number of discrete actions
    """

    n: int

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a random action uniformly from `[0, n)`.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        action : chex.Array
            int32 — Sampled action index.
        """
        return jax.random.randint(
            key, shape=(), minval=0, maxval=self.n, dtype=jnp.int32
        )


@dataclass(frozen=True)
class Box:
    """
    Continuous box observation space with scalar bounds.

    Parameters
    ----------
    low : float
        Lower bound (inclusive) applied to all elements
    high : float
        Upper bound (inclusive) applied to all elements
    shape : Tuple[int, ...]
        Shape of a single observation
    dtype : type
        Element dtype. Defaults to `jnp.uint8`
    """

    low: float
    high: float
    shape: Tuple[int, ...]
    dtype: type = jnp.uint8

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a random observation uniformly within `[low, high]`.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            `dtype[*shape]` — Sampled observation array
        """
        return jax.random.randint(
            key,
            shape=self.shape,
            minval=int(self.low),
            maxval=int(self.high) + 1,
            dtype=self.dtype,
        )
