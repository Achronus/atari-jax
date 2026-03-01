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

import json
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np


class AtariRenderer:
    """Manifest-aware display renderer. Zero training-path impact.

    Falls back to ``game.render(state)`` when no sprite PNG assets exist.
    Drop PNGs into ``assets/<game>/sprites/`` to enable sprite blitting.

    Parameters
    ----------
    game_name : str
        Lowercase game name, e.g. ``"breakout"``.
    scale : int (optional)
        Integer upscale factor applied to the native 160 × 210 screen.
        Default is ``3``.
    """

    def __init__(self, game_name: str, scale: int = 3) -> None:
        self.game_name = game_name
        self.scale = scale
        self._manifest = self._load_manifest()
        self._use_sprites = self._check_sprites()

    def _load_manifest(self) -> dict | None:
        path = Path(f"assets/{self.game_name}/manifest.json")
        return json.loads(path.read_text()) if path.exists() else None

    def _check_sprites(self) -> bool:
        d = Path(f"assets/{self.game_name}/sprites")
        return d.exists() and any(d.glob("*.png"))

    def render(self, state: Any, game: Any) -> np.ndarray:
        """
        Render a game frame, using sprites when available.

        Parameters
        ----------
        state : Any
            Current game state pytree (``AtariState`` subclass).
        game : Any
            The unwrapped ``AtariEnv`` instance.

        Returns
        -------
        frame : np.ndarray
            uint8[210, 160, 3] — RGB image as a NumPy array.
        """
        if self._use_sprites:
            return self._render_sprites(state)
        return np.asarray(game.render(state))

    def _render_sprites(self, state: Any) -> np.ndarray:
        raise NotImplementedError(
            f"Sprite blitting is not yet implemented. "
            f"Drop PNG assets into assets/{self.game_name}/sprites/ once available."
        )


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

    Uses `AtariRenderer` so that sprite assets (when present in
    ``assets/<game>/sprites/``) are automatically picked up without
    any code change.  Falls back to the JAX ``game.render(state)``
    path when no sprites exist.

    Keyboard controls vary by game; see the game's ``_key_map`` method
    or documentation page.  Press ``Esc`` or close the window to exit.

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
        Target agent steps per second.  Default is `15`, matching the
        ALE 60 Hz display rate at 4× frame skip.
    """
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "pygame is required for interactive play. "
            "Install it with: pip install pygame"
        ) from exc

    from atarax.games.registry import get_game
    from atarax.spec import EnvSpec

    spec = EnvSpec.parse(game_id)
    game = get_game(spec.env_name)()
    renderer = AtariRenderer(spec.env_name, scale=scale)

    reset_fn = jax.jit(game.reset)
    step_fn = jax.jit(game.step)

    key = jax.random.PRNGKey(42)
    obs, state = reset_fn(key)

    # Warm up XLA compilation before the game loop
    _wu, *_ = step_fn(state, jnp.int32(0))
    _wu.block_until_ready()

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

        obs, state, _, done, info = step_fn(state, jnp.int32(action))

        if bool(done):
            key, subkey = jax.random.split(key)
            obs, state = reset_fn(subkey)

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
