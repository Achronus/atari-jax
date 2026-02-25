# Time Pilot

> Game ID: `"atari/time_pilot-v0"`

Omnidirectional aerial combat through multiple historical eras. Your plane is always at the screen centre; enemies scroll relative to your heading. Clear 25 enemies to advance to the next era.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP (aim up) |
| `3` | RIGHT (aim right / clockwise) |
| `4` | DOWN (aim down) |
| `5` | LEFT (aim left / counter-clockwise) |
| `6` | UP+FIRE |
| `7` | RIGHT+FIRE |
| `8` | DOWN+FIRE |

## Reward

| Event | Reward |
| --- | --- |
| Era 1 (WWI) enemy destroyed | `+100` |
| Era 2 (WWII) enemy destroyed | `+150` |
| Era 3 (Korea) enemy destroyed | `+200` |
| Era 4 (Vietnam) enemy destroyed | `+300` |
| Era 5 (Present) enemy destroyed | `+500` |
| Boss destroyed | `+1000` |

## Episode End

The episode ends when all lives are lost. A life is lost when the player's plane is hit by an enemy bullet or by a colliding enemy aircraft.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player | fixed at x = 80, y = 105 (screen centre) |
| Enemies | distributed across the screen; world scrolls relative to heading |
| Enemies to clear per era | 25 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/time_pilot-v0")
play("atari/time_pilot-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP (aim up) |
| `→` / `D` | RIGHT (aim right) |
| `↓` / `S` | DOWN (aim down) |
| `←` / `A` | LEFT (aim left) |
| `Esc` / close window | Quit |
