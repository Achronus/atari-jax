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
from typing import Any, Dict, List, Optional, Tuple, Union

import chex
import jax.numpy as jnp

from atarax.core.state import AtariState
from atarax.env._kernels import (
    _jit_sample,
    _make_group_kernels,
    jit_reset,
    jit_reset_single,
    jit_rollout,
    jit_rollout_single,
    jit_step,
    jit_step_single,
)
from atarax.env.spaces import Box, Discrete
from atarax.games.registry import _GAMES, GAME_GROUPS, GAME_IDS, WARMUP_FRAMES_ARRAY
from atarax.utils.rom_loader import load_rom

_N_ACTIONS: int = 18
_VALID_MODES = ("all", "single", "group")


def _resolve_group(
    group: Union[str, List[str]],
    game_id: str,
) -> Tuple[int, ...]:
    """
    Resolve a group specification to a sorted tuple of absolute game IDs.

    Parameters
    ----------
    group : str | List[str]
        Predefined group name (e.g. `"atari5"`) or explicit list of ALE
        game names (e.g. `["breakout", "pong"]`).
    game_id : str
        The game being created; validated to be a member of the group.

    Returns
    -------
    group_game_ids : Tuple[int]
        Sorted tuple of absolute game IDs.

    Raises
    ------
    ValueError
        If the group name is unknown, a game name is invalid, or `game_id`
        is not a member of the group.
    """
    if isinstance(group, str):
        if group not in GAME_GROUPS:
            raise ValueError(
                f"Unknown group {group!r}. "
                f"Available predefined groups: {sorted(GAME_GROUPS)}"
            )
        game_names = GAME_GROUPS[group]
    else:
        game_names = list(group)

    unknown = [n for n in game_names if n not in GAME_IDS]
    if unknown:
        raise ValueError(
            f"Unknown game name(s) in group: {unknown}. "
            f"Available games: {sorted(GAME_IDS)}"
        )

    if game_id not in game_names:
        raise ValueError(
            f"game_id {game_id!r} is not a member of the specified group: "
            f"{sorted(game_names)}"
        )

    return tuple(sorted(GAME_IDS[n] for n in game_names))


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
    compile_mode : str (optional)
        Compilation strategy for the JIT kernels.
        - `"all"` (default): compiles all 57 game branches into a single XLA program via
        `jax.lax.switch`, so every game with the same ROM size shares one compilation.
        - `"single"`: makes `game_id` a static JIT argument, constant-folding the
        dispatch to only the selected game's branch — smaller programs and
        faster cold-start compilation at the cost of one XLA program per game.
        - `"group"`: compiles only the N games in the specified `group` via an N-way
        `jax.lax.switch`. Requires `group` to be provided.
    group : str or list of str (optional)
        Required when `compile_mode="group"`.  Either a predefined group name
        (`"atari5"`, `"atari10"`, `"atari26"`) or an explicit list of ALE
        game names.  The `game_id` must be a member of the group.
    """

    def __init__(
        self,
        game_id: str,
        params: EnvParams = EnvParams(),
        compile_mode: str = "all",
        group: Optional[Union[str, List[str]]] = None,
    ) -> None:
        if game_id not in GAME_IDS:
            raise ValueError(
                f"Unknown game_id {game_id!r}. Available games: {sorted(GAME_IDS)}"
            )
        if compile_mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid compile_mode {compile_mode!r}. Choose from {_VALID_MODES}."
            )
        if compile_mode == "group" and group is None:
            raise ValueError(
                "compile_mode='group' requires a 'group' argument "
                "(e.g. group='atari5' or group=['breakout', 'pong'])."
            )
        if compile_mode != "group" and group is not None:
            raise ValueError(
                f"'group' argument is only valid with compile_mode='group', "
                f"not {compile_mode!r}."
            )

        self._game_id = game_id
        self._params = params
        self._compile_mode = compile_mode
        self._game = _GAMES[GAME_IDS[game_id]].game
        self._rom = load_rom(game_id)
        self._game_id_jax = jnp.int32(GAME_IDS[game_id])
        self._game_id_int = int(GAME_IDS[game_id])
        self._warmup_frames = WARMUP_FRAMES_ARRAY[GAME_IDS[game_id]]

        if compile_mode == "group":
            group_game_ids = _resolve_group(group, game_id)
            self._group_kernels = _make_group_kernels(group_game_ids)
        else:
            self._group_kernels = None

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
        noop_max = jnp.int32(self._params.noop_max)

        if self._compile_mode == "single":
            return jit_reset_single(
                key, self._rom, self._game_id_int, self._warmup_frames, noop_max
            )
        if self._compile_mode == "group":
            return self._group_kernels.reset(
                key, self._rom, self._game_id_jax, self._warmup_frames, noop_max
            )
        return jit_reset(
            key, self._rom, self._game_id_jax, self._warmup_frames, noop_max
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
        fs = self._params.frame_skip
        me = self._params.max_episode_steps

        if self._compile_mode == "single":
            return jit_step_single(state, self._rom, action, self._game_id_int, fs, me)

        if self._compile_mode == "group":
            return self._group_kernels.step(
                state, self._rom, action, self._game_id_jax, fs, me
            )

        return jit_step(state, self._rom, action, self._game_id_jax, fs, me)

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
        fs = self._params.frame_skip
        me = self._params.max_episode_steps

        if self._compile_mode == "single":
            return jit_rollout_single(
                state, self._rom, actions, self._game_id_int, fs, me
            )

        if self._compile_mode == "group":
            return self._group_kernels.rollout(
                state, self._rom, actions, self._game_id_jax, fs, me
            )

        return jit_rollout(state, self._rom, actions, self._game_id_jax, fs, me)

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
