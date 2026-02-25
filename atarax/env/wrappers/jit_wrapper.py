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

"""JitWrapper — eagerly compiles `reset`, `step`, and `rollout` via `jax.jit`."""

import pathlib
from typing import Any, Dict, Tuple

import chex
import jax

from atarax.env._base import Env
from atarax.env._compile import DEFAULT_CACHE_DIR, setup_cache
from atarax.env.wrappers.base import Wrapper


class JitWrapper(Wrapper):
    """
    Wrap an `Env` so that `reset`, `step`, and `rollout` are compiled with
    `jax.jit` on construction.

    The compiled functions are cached per-instance, so the first call to
    `reset` or `step` pays the XLA compilation cost; all subsequent calls
    return compiled-kernel speed.  When used via `make(..., jit_compile=True)`,
    the warm-up call inside `make()` triggers compilation immediately.

    `JitWrapper` also configures the persistent XLA compilation cache so that
    compiled kernels survive across Python sessions.

    Parameters
    ----------
    env : Env
        Environment to wrap.  May itself be a wrapper stack; `JitWrapper`
        is always applied last (outermost) so the full transform pipeline
        is captured in a single compiled program.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.  Defaults to
        `~/.cache/atarax/xla_cache`.  Pass `None` to disable.
    """

    def __init__(
        self,
        env: Env,
        cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__(env)
        setup_cache(cache_dir)

        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)
        self._jit_rollout = jax.jit(env.rollout)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, Any]:
        """
        Reset the environment via the JIT-compiled kernel.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            First observation.
        state : Any
            Initial environment state.
        """
        return self._jit_reset(key)

    def step(
        self,
        state: Any,
        action: chex.Array,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment via the JIT-compiled kernel.

        Parameters
        ----------
        state : Any
            Current environment state.
        action : chex.Array
            int32 — Action index.

        Returns
        -------
        obs : chex.Array
            Observation after the step.
        new_state : Any
            Updated environment state.
        reward : chex.Array
            float32 — Reward for this step.
        done : chex.Array
            bool — `True` when the episode has ended.
        info : Dict[str, Any]
            Auxiliary diagnostic information.
        """
        return self._jit_step(state, action)

    def rollout(
        self,
        state: Any,
        actions: chex.Array,
    ) -> Tuple[Any, Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]]:
        """
        Run a multi-step rollout via the JIT-compiled scan kernel.

        Parameters
        ----------
        state : Any
            Initial environment state.
        actions : chex.Array
            int32[T] — Action sequence of length T.

        Returns
        -------
        final_state : Any
            State after all T steps.
        transitions : Tuple[chex.Array, ...]
            `(obs, reward, done, info)` each with a leading T dimension.
        """
        return self._jit_rollout(state, actions)
