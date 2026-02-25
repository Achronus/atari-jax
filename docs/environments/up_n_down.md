# Up 'n Down

> Game ID: `"atari/up_n_down-v0"`

Drive a jeep on a winding mountain road, collecting flags and running over or jumping over enemy vehicles. Collect all flags in a level to advance.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (unused / honk) |
| `2` | UP (jump) |
| `3` | RIGHT (accelerate) |
| `4` | DOWN (unused) |
| `5` | LEFT (brake) |

## Reward

| Event | Reward |
| --- | --- |
| Flag collected | `+100` |
| Enemy squashed (land on enemy while airborne) | `+200` |
| Level complete | `+1000` |

## Episode End

The episode ends when all lives are lost. A life is lost by colliding with an enemy vehicle while on the ground.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Road y | y = 140 |
| Ground level | y = 128 (road y − 12) |
| Player x | x = 30 (fixed; road scrolls) |
| Flags (world coords) | evenly spaced from x = 200 to x = 1400 |
| Speed range | 1.0–4.0 px/frame |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/up_n_down-v0")
play("atari/up_n_down-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | UP (jump) |
| `→` / `D` | RIGHT (accelerate) |
| `←` / `A` | LEFT (brake) |
| `Esc` / close window | Quit |
