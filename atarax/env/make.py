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

"""Factory functions for creating AtariEnv instances."""

import json
import pathlib
import re
import shutil
from typing import List, Type

import jax
import jax.numpy as jnp
from tqdm import tqdm

from atarax.env._compile import DEFAULT_CACHE_DIR, _live_bar, setup_cache
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.env import Env
from atarax.env.spec import EnvSpec
from atarax.env.vec_env import VecEnv
from atarax.env.wrappers import AtariPreprocessing, Wrapper
from atarax.env.wrappers.base import _WrapperFactory
from atarax.games.registry import GAME_IDS

_ENV_ID_RE = re.compile(r"^([^/]+)/(.+)-v(\d+)$")
_MISSING = object()


def _wrapper_str(w: Type[Wrapper] | _WrapperFactory) -> str:
    """Return a manifest-friendly string for a wrapper class or factory."""
    if isinstance(w, type):
        return w.__name__

    name = w._cls.__name__
    if w._kwargs:
        parts = ", ".join(f"{k}={v!r}" for k, v in sorted(w._kwargs.items()))
        return f"{name}({parts})"

    return name


def _resolve_spec(game_id: str | EnvSpec) -> str:
    """
    Return the ALE name from an `EnvSpec` or `"[engine]/[name]-v[N]"` string.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier.

    Returns
    -------
    ale_name : str
        Internal ALE game name (e.g. `"breakout"`).

    Raises
    ------
    id_error : ValueError
        If `game_id` is a string that does not match the
        `"[engine]/[name]-v[N]"` format.
    """
    if isinstance(game_id, EnvSpec):
        return game_id.env_name

    m = _ENV_ID_RE.match(game_id)

    if not m:
        raise ValueError(
            f"Invalid environment ID {game_id!r}. "
            "Use an EnvSpec or the format 'atari/<game_name>-v0', "
            "e.g. EnvSpec('atari', 'breakout') or 'atari/breakout-v0'."
        )

    return m.group(2)


def make(
    game_id: str | EnvSpec,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    show_compile_progress: bool = False,
) -> Env:
    """
    Create a single `AtariEnv`, optionally with wrappers applied.

    The returned environment is compiled with a single-game XLA program
    (``compile_mode="single"``), which traces only the selected game's
    dispatch branches and produces a smaller, faster-to-compile program
    than the full 57-game switch.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or the canonical string
        `"atari/breakout-v0"`.
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes or pre-configured factories applied innermost-first
        around the base env. Mutually exclusive with `preset`.
        Default is `None`.
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing` wrapper) from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`. Default is `False`.
    jit_compile : bool (optional)
        Eagerly trigger kernel compilation now rather than lazily on the
        first call.  `reset`, `step`, and `rollout` are always JIT-compiled
        via module-level kernels regardless of this flag.  Set to `False`
        only to defer compilation to first use.  To disable JIT for
        debugging, use `jax.disable_jit()` or set `JAX_DISABLE_JIT=1`.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.  Defaults to
        `~/.cache/atari-jax/xla_cache`. Pass `None` to disable.
    show_compile_progress : bool (optional)
        Display a spinner on the first (compilation) call of each method.
        Automatically suppressed when the game is already in the precompile
        manifest for this configuration.  Default is `False`.

    Returns
    -------
    env : Env
        Configured environment (bare `AtariEnv` or wrapped `Wrapper`).

    Raises
    ------
    wrapper_error : ValueError
        If both `wrappers` and `preset` are provided.
    id_error : ValueError
        If `game_id` is a string that does not match the
        `"[engine]/[name]-v[N]"` format.
    """
    ale_name = _resolve_spec(game_id)

    if wrappers is not None and preset:
        raise ValueError("Provide either `wrappers` or `preset`, not both.")

    setup_cache(cache_dir)

    env: Env = AtariEnv(ale_name, params or EnvParams(), compile_mode="single")

    if preset:
        env = AtariPreprocessing(env, h=84, w=84, n_stack=4)
    elif wrappers:
        for w in wrappers:
            env = w(env)

    if jit_compile:
        _key = jax.random.PRNGKey(0)
        _show = show_compile_progress and not (
            cache_dir is not None
            and _manifest_has_game(
                pathlib.Path(cache_dir), ale_name, 1, preset, wrappers
            )
        )
        if _show:
            with tqdm(total=1, desc="Compiling...", leave=True) as _bar:
                with _live_bar(_bar):
                    _, _state = env.reset(_key)
                    env.step(_state, env.sample(_key))
                _bar.update(1)
        else:
            _, _state = env.reset(_key)
            env.step(_state, env.sample(_key))

    return env


def make_vec(
    game_id: str | EnvSpec,
    n_envs: int,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    show_compile_progress: bool = False,
) -> VecEnv:
    """
    Create a `VecEnv` with `n_envs` parallel environments.

    In JAX, parallelism is achieved via `jax.vmap` rather than
    multiprocessing.  `VecEnv.reset(key)` splits the key into `n_envs`
    sub-keys so each environment starts from a distinct random state.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or the canonical string
        `"atari/breakout-v0"`.
    n_envs : int
        Number of parallel environments.
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`. Default is `None`.
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing` wrapper) from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`. Default is `False`.
    jit_compile : bool (optional)
        Eagerly trigger kernel compilation now rather than lazily on the
        first call.  All kernels are always JIT-compiled via module-level
        functions regardless of this flag.  Set to `False` only to defer
        compilation to first use.  To disable JIT for debugging, use
        `jax.disable_jit()` or set `JAX_DISABLE_JIT=1`.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.  Defaults to
        `~/.cache/atari-jax/xla_cache`.  Pass `None` to disable.
    show_compile_progress : bool (optional)
        Display a spinner on the first (compilation) call of each vmapped
        method.  Automatically suppressed when the game is already in the
        precompile manifest for this configuration.  Default is `False`.

    Returns
    -------
    vec_env : VecEnv
        Vectorised environment wrapping `n_envs` copies of the configured
        single-instance env.

    Raises
    ------
    wrapper_error : ValueError
        If both `wrappers` and `preset` are provided.
    id_error : ValueError
        If `game_id` is a string that does not match the
        `"[engine]/[name]-v[N]"` format.
    """
    ale_name = _resolve_spec(game_id)

    env = make(
        game_id,
        params=params,
        wrappers=wrappers,
        preset=preset,
        jit_compile=False,
        cache_dir=cache_dir,
    )

    vec_env = VecEnv(env, n_envs)

    if jit_compile:
        _key = jax.random.PRNGKey(0)
        _show = show_compile_progress and not (
            cache_dir is not None
            and _manifest_has_game(
                pathlib.Path(cache_dir), ale_name, n_envs, preset, wrappers
            )
        )
        if _show:
            with tqdm(total=1, desc="Compiling...", leave=True) as _bar:
                with _live_bar(_bar):
                    _, _states = vec_env.reset(_key)
                    vec_env.step(_states, vec_env.sample(_key))
                _bar.update(1)
        else:
            _, _states = vec_env.reset(_key)
            vec_env.step(_states, vec_env.sample(_key))

    return vec_env


_MANIFEST_NAME = "precompile_manifest.json"


def _config_entry(
    n_envs: int,
    n_steps: int | None,
    preset: bool,
    wrappers: List[Type] | None,
) -> dict:
    return {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "preset": preset,
        "wrappers": [_wrapper_str(w) for w in wrappers] if wrappers else None,
    }


def _manifest_has_game(
    cache_dir: pathlib.Path,
    ale_name: str,
    n_envs: int,
    preset: bool,
    wrappers,
    *,
    n_steps=_MISSING,
) -> bool:
    """
    Return `True` if `ale_name` appears in the compiled list for a matching config.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory containing the manifest file.
    ale_name : str
        ALE game name to look up (e.g. `"breakout"`).
    n_envs : int
        Must match the entry's `n_envs` field.
    preset : bool
        Must match the entry's `preset` field.
    wrappers : list or None
        Wrapper list (classes / factories); serialised for comparison.
    n_steps : int | None (optional)
        When provided (including `None`), the entry's `n_steps` must also
        match.  Omit to ignore `n_steps` (used by `make()` / `make_vec()`
        where rollout length does not affect reset/step compilation).

    Returns
    -------
    found : bool
        `True` if a matching entry contains `ale_name` in its
        `"compiled"` list.
    """
    p = cache_dir / _MANIFEST_NAME
    if not p.exists():
        return False
    try:
        configs = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    wrappers_str = [_wrapper_str(w) for w in wrappers] if wrappers else None
    for entry in configs:
        n_steps_ok = n_steps is _MISSING or entry.get("n_steps") == n_steps
        if (
            entry.get("n_envs") == n_envs
            and n_steps_ok
            and entry.get("preset") == preset
            and entry.get("wrappers") == wrappers_str
            and ale_name in entry.get("compiled", [])
        ):
            return True
    return False


def _manifest_add_game(
    cache_dir: pathlib.Path,
    ale_name: str,
    n_envs: int,
    n_steps: int | None,
    preset: bool,
    wrappers,
) -> None:
    """
    Append `ale_name` to the `"compiled"` list for the matching config entry.

    Creates the entry if it does not exist.  Writes to disk immediately so
    that an interrupted `precompile_all()` run can be resumed.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory containing the manifest file.
    ale_name : str
        ALE game name to record (e.g. `"breakout"`).
    n_envs : int
        Entry key field.
    n_steps : int | None
        Entry key field.
    preset : bool
        Entry key field.
    wrappers : list | None
        Entry key field; serialised via `_wrapper_str`.
    """
    p = cache_dir / _MANIFEST_NAME
    wrappers_str = [_wrapper_str(w) for w in wrappers] if wrappers else None
    configs: list = []
    if p.exists():
        try:
            configs = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    for entry in configs:
        if (
            entry.get("n_envs") == n_envs
            and entry.get("n_steps") == n_steps
            and entry.get("preset") == preset
            and entry.get("wrappers") == wrappers_str
        ):
            compiled = entry.setdefault("compiled", [])
            if ale_name not in compiled:
                compiled.append(ale_name)
            break
    else:
        configs.append(
            {
                "n_envs": n_envs,
                "n_steps": n_steps,
                "preset": preset,
                "wrappers": wrappers_str,
                "compiled": [ale_name],
            }
        )
    p.write_text(json.dumps(configs, indent=2))


def precompile_all(
    *,
    n_envs: int = 1,
    n_steps: int | None = None,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    clear_cache: bool = False,
) -> None:
    """
    Compile and cache all 57 game environments.

    Runs `reset()`, `step()`, and optionally `rollout()` for every game so
    that JAX traces and compiles the full emulation graph and writes the result
    to the XLA persistent cache.  Subsequent `make()` / `make_vec()` calls for
    any game will load from cache rather than recompiling.

    Multiple configurations can be cached side-by-side.  Each successfully
    compiled game is appended to the `"compiled"` list in
    `precompile_manifest.json` immediately, so an interrupted run can be
    resumed — already-compiled games are skipped automatically.

    The cache key depends on the exact input shapes and computation graph, so
    `n_envs`, `n_steps`, `wrappers`, and `preset` must all match how the
    environments will be used in your training loop for cache hits to occur.

    Parameters
    ----------
    n_envs : int (optional)
        Number of parallel environments to compile for.  `1` (default) uses
        `make()` and caches single-env `reset` / `step` shapes.  Any value
        greater than `1` uses `make_vec()` and caches batched shapes.
    n_steps : int (optional)
        If provided, also compiles and caches the rollout function for a
        fixed sequence length of `n_steps`.  This corresponds to `T` in
        the `actions` array passed to `rollout()`.  Leave as `None` to skip
        rollout precompilation. Default is `None`.
    params : EnvParams (optional)
        Shared environment parameters. Defaults to `EnvParams()`.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`. Default is `None`.
    preset : bool (optional)
        Apply the DQN preprocessing stack to every environment.
        Mutually exclusive with `wrappers`. Default is `False`.
    cache_dir : Path | str | None (optional)
        Cache directory.  Defaults to `~/.cache/atari-jax/xla_cache`.
        Pass `None` to disable caching (compilation still occurs but is not
        stored to disk).
    clear_cache : bool (optional)
        Delete the entire cache directory before compiling, forcing a full
        recompile of all 57 games and clearing all previously stored
        configurations.  Default is `False`.
    """
    if cache_dir is not None:
        cache_path = pathlib.Path(cache_dir)
        if clear_cache and cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)

    setup_cache(cache_dir)

    game_names = list(GAME_IDS.keys())
    key = jax.random.PRNGKey(0)

    print(
        "Hold tight! Compiling and caching all 57 Atari environments. This may take a while..."
    )
    with tqdm(game_names, unit="env") as bar:
        for game_name in bar:
            spec = EnvSpec("atari", game_name)

            if cache_dir is not None and _manifest_has_game(
                cache_path, game_name, n_envs, preset, wrappers, n_steps=n_steps
            ):
                bar.set_description(f"Skipping {spec.id} (cached)")
                continue

            bar.set_description(f"Compiling {spec.id}")

            with _live_bar(bar):
                if n_envs == 1:
                    env = make(
                        spec,
                        params=params,
                        wrappers=wrappers,
                        preset=preset,
                        jit_compile=True,
                        cache_dir=None,
                    )
                    _, state = env.reset(key)
                    env.step(state, env.sample(key))

                    if n_steps is not None:
                        env.rollout(state, jnp.zeros(n_steps, dtype=jnp.int32))
                else:
                    vec_env = make_vec(
                        spec,
                        n_envs=n_envs,
                        params=params,
                        wrappers=wrappers,
                        preset=preset,
                        jit_compile=True,
                        cache_dir=None,
                    )
                    _, states = vec_env.reset(key)
                    vec_env.step(states, jnp.zeros(n_envs, dtype=jnp.int32))

                    if n_steps is not None:
                        vec_env.rollout(
                            states, jnp.zeros((n_envs, n_steps), dtype=jnp.int32)
                        )

            if cache_dir is not None:
                _manifest_add_game(
                    cache_path, game_name, n_envs, n_steps, preset, wrappers
                )
