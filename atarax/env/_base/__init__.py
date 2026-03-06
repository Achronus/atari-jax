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

"""Abstract base classes for atarax game templates.

Provides one state dataclass and one game base class per template:

- `FixedShooterState` / `FixedShooterGame` — T1 (Space Invaders, Phoenix, …)
- `Free2DShooterState` / `Free2DShooterGame` — T2 (Asteroids, Gravitar, …)
- `MazeNavigatorState` / `MazeNavigatorGame` — T5 (Pac-Man, …)
- `BallPhysicsState` / `BallPhysicsGame` — T6 (Breakout, Video Pinball, …)
"""

from atarax.env._base.ball_physics import BallPhysicsGame, BallPhysicsState
from atarax.env._base.fixed_shooter import FixedShooterGame, FixedShooterState
from atarax.env._base.free_2d_shooter import Free2DShooterGame, Free2DShooterState
from atarax.env._base.maze_navigator import MazeNavigatorGame, MazeNavigatorState

__all__ = [
    "BallPhysicsState",
    "BallPhysicsGame",
    "FixedShooterState",
    "FixedShooterGame",
    "Free2DShooterState",
    "Free2DShooterGame",
    "MazeNavigatorState",
    "MazeNavigatorGame",
]
