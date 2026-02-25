# Gravitar

> Game ID: `"atari/gravitar-v0"`

Navigate a gravity-filled star system, rescuing astronauts from planet surfaces while fighting bunkers and fuel pods. Manage fuel carefully — running out costs a life.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(7)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | THRUST |
| `3` | RIGHT (rotate clockwise) |
| `4` | LEFT (rotate counter-clockwise) |
| `5` | SHIELD |
| `6` | THRUST + FIRE |

## Reward

| Event | Reward |
| --- | --- |
| Bunker destroyed | +250 |
| Fuel pod destroyed | +500 |
| Astronaut rescued | +1000 |

## Episode End

The episode ends when all lives are lost. A life is lost by crashing into the ground without the shield active, or by running out of fuel.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Ground | y = 180 |
| Bunker y | y = 175 |
| Bunker x positions | x = 25, 55, 80, 105, 135 |
| Astronaut y | y = 170 |
| Astronaut x positions | x = 30, 65, 95, 130 |
| Fuel bar | top 8 rows (width proportional to fuel) |
| Starting fuel | 700 units |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/gravitar-v0")
play("atari/gravitar-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | THRUST |
| `→` / `D` | RIGHT (rotate clockwise) |
| `←` / `A` | LEFT (rotate counter-clockwise) |
| `Left Shift` | SHIELD |
| `Esc` / close window | Quit |
