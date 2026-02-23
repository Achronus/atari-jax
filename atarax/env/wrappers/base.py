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

from abc import abstractmethod
from typing import Any, Dict, Self, Tuple, overload

import chex

from atarax.core.state import AtariState
from atarax.env.env import Env
from atarax.env.spaces import Box, Discrete


class _WrapperFactory:
    """
    Deferred wrapper returned by `Wrapper.__new__` when called without an `env`.

    Calling the factory with an `env` creates the intended wrapper with the
    pre-bound keyword arguments.

    Parameters
    ----------
    cls : type
        Concrete `Wrapper` subclass to instantiate.
    **kwargs
        Keyword arguments forwarded to `cls.__init__` when the factory is called.
    """

    __slots__ = ("_cls", "_kwargs")

    def __init__(self, cls: type, **kwargs) -> None:
        self._cls = cls
        self._kwargs = kwargs

    def __call__(self, env: Env) -> "Wrapper":
        """
        Wrap `env` using the stored class and keyword arguments.

        Parameters
        ----------
        env : Env
            Environment to wrap.

        Returns
        -------
        wrapper : Wrapper
            Configured wrapper instance.
        """
        return self._cls(env, **self._kwargs)


class Wrapper(Env):
    """
    Abstract base class for AtariEnv wrappers.

    Subclasses must implement `reset` and `step`. The `sample`,
    `observation_space`, and `action_space` members delegate to the
    inner environment by default and may be overridden when the wrapper
    changes the observation shape or action set.

    `rollout` is inherited from `Env` and uses `jax.lax.scan(self.step, ...)`,
    so all observation and reward transforms are applied at every step.

    Parameterised wrappers support a **factory mode**: calling the class
    without an `env` (using only keyword arguments) returns a
    `_WrapperFactory` rather than a live wrapper.  The factory can later
    be called with an `env` to create the proper instance, making it
    usable inside the `wrappers=[...]` list passed to `make()`:

    .. code-block:: python

        make("atari/breakout-v0", wrappers=[
            GrayscaleObservation,          # bare class
            ResizeObservation(h=64, w=64), # pre-configured factory
        ])

    Parameters
    ----------
    env : Env
        Inner environment to wrap.
    """

    @overload
    def __new__(cls, env: None = ..., **kwargs) -> "_WrapperFactory": ...

    @overload
    def __new__(cls, env: Env, **kwargs) -> Self: ...

    def __new__(cls, env=None, **kwargs):
        if env is None:
            factory = object.__new__(_WrapperFactory)
            _WrapperFactory.__init__(factory, cls, **kwargs)
            return factory

        return super().__new__(cls)

    def __init__(self, env: Env) -> None:
        self._env = env

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._env!r}>"

    @abstractmethod
    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset the environment and return the initial observation and state.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — First RGB observation.
        state : AtariState
            Initial machine state after reset and no-ops.
        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self, state, action: chex.Array
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        state : AtariState
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            `uint8[210, 160, 3]` — Observation after the step
        new_state : AtariState
            Updated machine state
        reward : chex.Array
            float32 — Total reward accumulated over skipped frames
        done : chex.Array
            bool — `True` when the episode has ended
        info : Dict[str, Any]
            `{"lives": int32, "episode_frame": int32, "truncated": bool}`
        """
        raise NotImplementedError()

    @property
    def unwrapped(self) -> "Env":
        """Return the innermost `AtariEnv` by delegating through the wrapper chain."""
        return self._env.unwrapped

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a uniformly-random action from the action space.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key
        """
        return self._env.sample(key)

    @property
    def observation_space(self) -> Box:
        """Environment's observation space."""
        return self._env.observation_space

    @property
    def action_space(self) -> Discrete:
        """Environment's action space."""
        return self._env.action_space
