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
from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from atarax.core.state import AtariState
from atarax.env._kernels import _jit_sample, jit_reset, jit_rollout, jit_step
from atarax.env.spaces import Box, Discrete
from atarax.games.registry import GAME_IDS, WARMUP_FRAMES_ARRAY, _GAMES
from atarax.utils.rom_loader import load_rom

_N_ACTIONS: int = 18


@dataclass(frozen=True)
class EnvParams:
    """
    Hyper-parameters for `AtariEnv`.

    Parameters
    ----------
    noop_max : int
        Maximum number of NOOP actions sampled uniformly at episode start.
        Set to 0 to disable random starts.
    frame_skip : int
        Number of emulator frames to repeat each action. Reward is summed
        across skipped frames.
    max_episode_steps : int
        Hard episode length limit. Terminal is forced when
        `episode_frame >= max_episode_steps`. Default matches the ALE
        standard of 108 000 frames at 4× frame skip.
    """

    noop_max: int = 30
    frame_skip: int = 4
    max_episode_steps: int = 27000


class AtariEnv:
    """
    Gymnax-style Atari 2600 environment.

    The environment is fully `jit` / `vmap` / `lax.scan` compatible. All state is
    captured in `AtariState`.

    Parameters
    ----------
    game_id : str
        ALE game identifier (e.g. `"breakout"`, `"pong"`)
    params : EnvParams
        Environment hyper-parameters. Defaults to `EnvParams()`
    """

    def __init__(self, game_id: str, params: EnvParams = EnvParams()) -> None:
        if game_id not in GAME_IDS:
            raise ValueError(
                f"Unknown game_id {game_id!r}. Available games: {sorted(GAME_IDS)}"
            )

        self._game_id = game_id
        self._params = params
        self._game = _GAMES[GAME_IDS[game_id]].game
        self._rom = load_rom(game_id)
        self._game_id_jax = jnp.int32(GAME_IDS[game_id])
        self._warmup_frames = WARMUP_FRAMES_ARRAY[GAME_IDS[game_id]]

    @property
    def default_params(self) -> EnvParams:
        """Get the environments parameters."""
        return self._params

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset the environment and return the first observation.

        Runs the game-specific warmup frames (60 by default) then executes
        a uniformly-sampled number of NOOP actions in `[0, noop_max]` to
        provide stochastic starting positions.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key used to sample the number of no-op steps.

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — First RGB observation.
        state : AtariState
            Initial machine state after reset and no-ops.
        """
        return jit_reset(
            key,
            self._rom,
            self._game_id_jax,
            self._warmup_frames,
            jnp.int32(self._params.noop_max),
        )

    def step(
        self,
        state: AtariState,
        action: chex.Array,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one RL step.

        Repeats `action` for `frame_skip` emulator frames and accumulates
        the reward. The episode terminates when the game logic signals
        terminal *or* `max_episode_steps` is reached.

        Parameters
        ----------
        state : AtariState
            Current machine state.
        action : chex.Array
            int32 — ALE action index (0–17).

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
        return jit_step(
            state,
            self._rom,
            action,
            self._game_id_jax,
            self._params.frame_skip,
            self._params.max_episode_steps,
        )

    def rollout(
        self,
        state: AtariState,
        actions: chex.Array,
    ) -> Tuple[AtariState, Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]]:
        """
        Run a compiled multi-step rollout.

        Parameters
        ----------
        state : AtariState
            Initial machine state.
        actions : chex.Array
            int32[T] — Action sequence; T determines the number of RL steps.

        Returns
        -------
        final_state : AtariState
            State after all T steps.
        transitions : tuple
            `(obs, reward, done, info)` each with a leading T dimension.
        """
        return jit_rollout(
            state,
            self._rom,
            actions,
            self._game_id_jax,
            self._params.frame_skip,
            self._params.max_episode_steps,
        )

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a uniformly-random action from the action space.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        action : chex.Array
            int32 — Random action index in `[0, 18)`.
        """
        return _jit_sample(key)

    @property
    def observation_space(self) -> Box:
        """Raw RGB observation space: `uint8[210, 160, 3]`."""
        return Box(low=0.0, high=255.0, shape=(210, 160, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        """Discrete action space: 18 ALE actions."""
        return Discrete(n=_N_ACTIONS)

    @property
    def num_actions(self) -> int:
        """Number of discrete actions (18)."""
        return _N_ACTIONS
