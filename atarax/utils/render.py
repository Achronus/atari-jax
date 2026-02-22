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

from typing import TYPE_CHECKING

import numpy as np

from atarax.core.state import AtariState

if TYPE_CHECKING:
    from atarax.env.spec import EnvSpec


def render(
    state: AtariState,
    *,
    scale: int = 2,
    caption: str = "atari-jax",
) -> None:
    """
    Display the current frame in a pygame window.

    Lazy-imports pygame so the library remains optional; install it with
    `pip install pygame` when human rendering is needed.

    Parameters
    ----------
    state : AtariState
        Current machine state.  Uses `state.screen` (uint8[210, 160, 3]).
    scale : int (optional)
        Integer upscale factor applied to the native 160 x 210 pixel screen. Default is `2`
    caption : str (optional)
        Window title shown in the title bar. Default is `atari-jax`
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

    screen_np = np.asarray(state.screen)  # (210, 160, 3) uint8
    h, w = screen_np.shape[:2]

    display = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption(caption)

    # pygame.surfarray.make_surface expects (width, height, 3)
    surf = pygame.surfarray.make_surface(screen_np.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, (w * scale, h * scale))
    display.blit(scaled, (0, 0))
    pygame.display.flip()


def play(
    game_id: "str | EnvSpec",
    *,
    scale: int = 3,
    fps: int = 60,
) -> None:
    """
    Play a game interactively in a pygame window using the ALE reference
    implementation.

    Backed by ale-py so the window opens instantly with no JAX compilation.
    Close the window or press `Esc` to quit.

    Requires `pygame` and `ale-py`.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or `"atari/breakout-v0"`.
    scale : int (optional)
        Integer upscale factor applied to the native 160x210 screen.
        Default is `3`, giving a 480x630 window.
    fps : int (optional)
        Target frames per second for the game loop. Default is `60`

    Notes
    -----
    Keyboard controls:

    - Arrow keys / `W A S D` — movement
    - `Space` — fire
    - `Esc` / close window — quit
    """
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "pygame is required for interactive play. "
            "Install it with: pip install pygame"
        ) from exc

    try:
        from ale_py import ALEInterface
        from ale_py.roms import get_rom_path
    except ImportError as exc:
        raise ImportError(
            "ale-py is required for interactive play. "
            "Install it with: pip install ale-py"
        ) from exc

    from atarax.env.make import _resolve_spec

    ale_name = _resolve_spec(game_id)

    ale = ALEInterface()
    ale.setInt("random_seed", 42)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("display_screen", False)
    ale.loadROM(get_rom_path(ale_name))  # type: ignore

    pygame.init()
    display = pygame.display.set_mode((160 * scale, 210 * scale))
    pygame.display.set_caption(f"atari-jax \u2014 {game_id}")
    clock = pygame.time.Clock()

    key_map = {
        pygame.K_UP: 2,
        pygame.K_w: 2,
        pygame.K_RIGHT: 3,
        pygame.K_d: 3,
        pygame.K_LEFT: 4,
        pygame.K_a: 4,
        pygame.K_DOWN: 5,
        pygame.K_s: 5,
        pygame.K_SPACE: 1,
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = next((a for k, a in key_map.items() if keys[k]), 0)

        ale.act(action)

        if ale.game_over():
            ale.reset_game()

        screen = ale.getScreenRGB()  # uint8[210, 160, 3]
        surf = pygame.surfarray.make_surface(screen.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surf, (160 * scale, 210 * scale))
        display.blit(scaled, (0, 0))
        pygame.display.flip()

        clock.tick(fps)

    pygame.quit()
