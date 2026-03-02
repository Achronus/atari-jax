# Gravitar

> Game ID: `"atari/gravitar-v0"`

Navigate a gravity-filled star system, rescuing astronauts from planet surfaces while fighting bunkers and fuel pods. Manage fuel carefully — running out costs a life.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP — thrust |
| `3` | RIGHT — rotate clockwise |
| `4` | LEFT — rotate counter-clockwise |
| `5` | DOWN — shield |
| `6` | UPRIGHT — thrust + rotate CW |
| `7` | UPLEFT — thrust + rotate CCW |
| `8` | DOWNRIGHT — shield + rotate CW |
| `9` | DOWNLEFT — shield + rotate CCW |
| `10` | UPFIRE — thrust + fire |
| `11` | RIGHTFIRE — rotate CW + fire |
| `12` | LEFTFIRE — rotate CCW + fire |
| `13` | DOWNFIRE — shield + fire |
| `14` | UPRIGHTFIRE — thrust + rotate CW + fire |
| `15` | UPLEFTFIRE — thrust + rotate CCW + fire |
| `16` | DOWNRIGHTFIRE — shield + rotate CW + fire |
| `17` | DOWNLEFTFIRE — shield + rotate CCW + fire |

## Reward

| Event | Reward |
| --- | --- |
| Bunker destroyed | +250 |
| Fuel pod destroyed | +500 |
| Astronaut rescued | +1000 |

## Episode End

The episode ends when all lives are lost. A life is lost by crashing into the ground without the shield active, or by running out of fuel.

## Lives

The player starts with 6 lives.

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
