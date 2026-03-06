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

import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np


class AtariRenderer:
    """Thin wrapper that delegates rendering to the game's procedural SDF renderer.

    All games produce frames via pure-JAX SDF rendering — no sprite assets are
    used or required. This class exists as a stable public API for tooling that
    needs a named renderer object (e.g. `play()`).

    Parameters
    ----------
    game_name : str
        Lowercase game name, e.g. `"breakout"`.
    scale : int (optional)
        Integer upscale factor applied to the native 160 × 210 screen.
        Default is `3`.
    """

    def __init__(self, game_name: str, scale: int = 3) -> None:
        self.game_name = game_name
        self.scale = scale

    def render(self, state: Any, game: Any) -> np.ndarray:
        """
        Render a game frame via the game's procedural SDF renderer.

        Calls `game.render(state)` and converts the JAX array to NumPy.

        Parameters
        ----------
        state : Any
            Current game state pytree (`AtariState` subclass).
        game : Any
            The unwrapped game instance (`AtaraxGame` subclass).

        Returns
        -------
        frame : np.ndarray
            `uint8[210, 160, 3]` — RGB image as a NumPy array.
        """
        return np.asarray(game.render(state))


def render(
    frame: chex.Array,
    *,
    scale: int = 2,
    caption: str = "atari-jax",
) -> None:
    """
    Display an RGB frame in a pygame window.

    Lazy-imports pygame so the library remains optional; install it with
    `pip install pygame` when human rendering is needed.

    Parameters
    ----------
    frame : chex.Array
        uint8[210, 160, 3] — RGB frame returned by `game.render(state)`.
    scale : int (optional)
        Integer upscale factor applied to the native 160 x 210 pixel screen.
        Default is `2`.
    caption : str (optional)
        Window title shown in the title bar. Default is `atari-jax`.
    """
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "pygame is required for human rendering. "
            "Install it with: pip install pygame"
        ) from exc

    if not pygame.get_init():
        pygame.init()

    frame_np = np.asarray(frame)  # (210, 160, 3) uint8
    h, w = frame_np.shape[:2]

    display = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption(caption)

    surf = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, (w * scale, h * scale))
    display.blit(scaled, (0, 0))
    pygame.display.flip()


def play(
    game_id: str,
    *,
    scale: int = 3,
    fps: int = 15,
) -> None:
    """
    Play a game interactively in a pygame window.

    Renders via the game's procedural SDF renderer. Keyboard controls vary by
    game; see the game's `_key_map` method or documentation page. Press `Esc`
    or close the window to exit.

    Requires `pygame` (`pip install pygame`).

    Parameters
    ----------
    game_id : str
        Environment ID in Gymnasium format, e.g. `"atari/breakout-v0"`
        (case-insensitive for engine and game name).
    scale : int (optional)
        Integer upscale factor applied to the native 160 × 210 screen.
        Default is `3`, giving a 480 × 630 window.
    fps : int (optional)
        Target agent steps per second. Default is `15`.
    """
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "pygame is required for interactive play. "
            "Install it with: pip install pygame"
        ) from exc

    from atarax.env.registry import PARAMS, get_game
    from atarax.game import AtaraxParams
    from atarax.spec import EnvSpec

    spec = EnvSpec.parse(game_id)
    game = get_game(game_id)()
    params = PARAMS.get(spec.env_name, AtaraxParams)()
    renderer = AtariRenderer(spec.env_name, scale=scale)

    reset_fn = jax.jit(game.reset)
    step_fn = jax.jit(game.step)

    key = jax.random.PRNGKey(42)
    obs, state = reset_fn(key, params)

    # Warm up XLA compilation before the game loop
    key, warmup_key = jax.random.split(key)
    _wu_obs, *_ = step_fn(warmup_key, state, jnp.int32(0), params)
    _wu_obs.block_until_ready()

    pygame.init()
    display = pygame.display.set_mode((160 * scale, 210 * scale))
    pygame.display.set_caption(f"atari-jax \u2014 {spec.env_name}")
    clock = pygame.time.Clock()
    key_map = game._key_map()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = next((a for k, a in key_map.items() if keys[k]), 0)

        key, step_key = jax.random.split(key)
        obs, state, _, done, info = step_fn(step_key, state, jnp.int32(action), params)

        if bool(done):
            key, subkey = jax.random.split(key)
            obs, state = reset_fn(subkey, params)

        frame_np = renderer.render(state, game)  # uint8[210, 160, 3]
        surf = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surf, (160 * scale, 210 * scale))
        display.blit(scaled, (0, 0))

        # Score overlay
        font = pygame.font.SysFont("monospace", max(12, 5 * scale), bold=True)
        score_val = int(state.score)
        lives_val = int(state.lives)
        label = (
            f"SCORE {score_val}"
            if lives_val == 0
            else f"SCORE {score_val}  LIVES {lives_val}"
        )
        text_surf = font.render(label, True, (255, 255, 255), (0, 0, 0))
        display.blit(text_surf, (4, 4))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def render_grid(frames: np.ndarray, nrow: int = 4) -> np.ndarray:
    """Tile N rendered frames into a `(nrow × ncol)` image grid.

    Useful for monitoring multiple parallel environments at a glance — pass the
    vmap-output frames from `jax.vmap(env.render)(states)` converted to numpy.

    Parameters
    ----------
    frames : np.ndarray
        `uint8[N, 210, 160, 3]` — One frame per environment.
    nrow : int
        Number of rows in the output grid. The number of columns is
        `ceil(N / nrow)`. Surplus grid slots are filled with black.

    Returns
    -------
    grid : np.ndarray
        `uint8[210 * nrow, 160 * ncol, 3]` — Tiled image.

    Examples
    --------
    >>> import jax, numpy as np
    >>> from atarax import make
    >>> from atarax.render import render_grid
    >>> env, params = make("atari/space_invaders-v0")
    >>> rngs = jax.random.split(jax.random.PRNGKey(0), 16)
    >>> states, _ = jax.vmap(env.reset, in_axes=(0, None))(rngs, params)
    >>> frames = np.asarray(jax.vmap(env.render)(states))
    >>> grid = render_grid(frames, nrow=4)  # (840, 640, 3)
    """
    N = frames.shape[0]
    ncol = math.ceil(N / nrow)
    pad = nrow * ncol - N
    if pad > 0:
        frames = np.concatenate(
            [frames, np.zeros((pad, 210, 160, 3), dtype=np.uint8)], axis=0
        )
    # (nrow*ncol, 210, 160, 3) → (nrow, ncol, 210, 160, 3)
    # → transpose to (nrow, 210, ncol, 160, 3) → (nrow*210, ncol*160, 3)
    grid = frames.reshape(nrow, ncol, 210, 160, 3)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(nrow * 210, ncol * 160, 3)
    return grid


def record_episode(
    game: Any,
    params: Any,
    policy_fn: Callable,
    *,
    seed: int = 0,
    path: str = "episode.gif",
    fps: int = 30,
    max_steps: int = 10_000,
) -> None:
    """Record a single episode to a gif or mp4 file.

    Runs the game under `policy_fn`, captures every rendered frame, and
    writes the result using `imageio`.  Requires `pip install imageio
    imageio-ffmpeg`.

    Parameters
    ----------
    game : AtaraxGame
        Unwrapped game instance (has `reset`, `step`, and `render`).
    params : AtaraxParams
        Environment parameters.
    policy_fn : Callable
        `(obs: jax.Array) -> int` — Maps an observation to an action.
        Pass a lambda wrapping your trained policy, or use
        `lambda obs: env.action_space.sample()` for a random policy.
    seed : int
        PRNG seed for the episode. Default `0`.
    path : str
        Output file path. Extension determines format: `.gif` or `.mp4`.
        Default `"episode.gif"`.
    fps : int
        Frames per second of the output video. Default `30`.
    max_steps : int
        Hard cap on episode length. Default `10_000`.

    Examples
    --------
    >>> from atarax.env.games.breakout import Breakout, BreakoutParams
    >>> from atarax.render import record_episode
    >>> game = Breakout()
    >>> record_episode(game, BreakoutParams(), lambda obs: 0, path="breakout.gif")
    """
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for record_episode. "
            "Install with: pip install imageio imageio-ffmpeg"
        ) from exc

    rng = jax.random.PRNGKey(seed)
    obs, state = game.reset(rng, params)
    frames = [np.asarray(game.render(state))]

    for _ in range(max_steps):
        rng, step_rng = jax.random.split(rng)
        action = policy_fn(obs)
        obs, state, _reward, done, _info = game.step(step_rng, state, action, params)
        frames.append(np.asarray(game.render(state)))
        if bool(done):
            break

    imageio.mimsave(path, frames, fps=fps)


def game_snapshot(game_id: str, *, seed: int = 0) -> np.ndarray:
    """Return frame-0 for a single registered game.

    Parameters
    ----------
    game_id : str
        Snake-case game name, e.g. `"breakout"`.
    seed : int
        PRNG seed. Default `0`.

    Returns
    -------
    frame : np.ndarray
        `uint8[210, 160, 3]` — RGB frame.
    """
    from atarax.env.registry import GAMES, PARAMS
    from atarax.game import AtaraxParams

    params = PARAMS.get(game_id, AtaraxParams)(noop_max=0)
    game = GAMES[game_id]()
    _, state = game.reset(jax.random.PRNGKey(seed), params)
    return np.asarray(game.render(state))


def snapshot_all_envs(
    game_ids: list[str] | None = None,
    *,
    seed: int = 0,
    nrow: int = 4,
) -> np.ndarray:
    """Return a tiled grid of frame-0 for every registered game.

    Loops :func:`game_snapshot` over all given games and tiles the results
    with :func:`render_grid`.  Useful for quick visual inspection during
    development.  The full 57-game collection image is deferred to the
    README once all games are implemented.

    Parameters
    ----------
    game_ids : list[str] | None
        Snake-case names, e.g. `["breakout", "pong"]`.
        Defaults to all games currently in `GAMES`.
    seed : int
        PRNG seed passed to every `reset` call. Default `0`.
    nrow : int
        Rows in the output grid. Default `4`.

    Returns
    -------
    grid : np.ndarray
        `uint8[210*nrow, 160*ncol, 3]` — tiled RGB image.

    Examples
    --------
    >>> from atarax.render import snapshot_all_envs
    >>> import imageio
    >>> grid = snapshot_all_envs(seed=0, nrow=4)
    >>> imageio.imwrite("all_envs.png", grid)
    """
    from atarax.env.registry import GAMES

    if game_ids is None:
        game_ids = list(GAMES.keys())

    frames = [game_snapshot(gid, seed=seed) for gid in game_ids]
    return render_grid(np.stack(frames, axis=0), nrow=nrow)
