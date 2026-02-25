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

import chex
import numpy as np


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

    Delegates to `AtariEnv.play()` on the game resolved from `game_id`.
    Keyboard controls vary by game; see the game's `_key_map` method
    or documentation page.

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
    from atarax.env.spec import EnvSpec
    from atarax.games.registry import get_game

    spec = EnvSpec.parse(game_id)
    game = get_game(spec.env_name)()
    game.play(scale=scale, fps=fps)
