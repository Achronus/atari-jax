# Bowling

> Game ID: `"atari/bowling-v0"`

Roll a bowling ball to knock down 10 pins at the far end of a lane.  The
player can curve the ball left or right before release.  Each set of 10 pins
is one frame; the episode ends after 10 frames.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(4)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP / hold |
| `1` | FIRE — release ball |
| `2` | LEFT — curve ball left or step left |
| `3` | RIGHT — curve ball right or step right |

## Reward

Points are awarded based on pins knocked down per roll.  Strikes and spares
earn bonus points.

| Result | Points |
| --- | --- |
| Strike (all 10, first ball) | pins + 10 bonus = 20 |
| Spare (all 10, second ball) | pins + 5 bonus = 15 |
| Open frame | number of pins knocked down |

## Episode End

The episode ends after 10 bowling frames are completed.

## Lives

No lives system.  `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Lane | x ∈ [60, 100), full height |
| Gutters | x ∈ [54, 60) and x ∈ [100, 106) |
| Pins | 10 pins in triangle at y ≈ 30, centred at x = 80 |
| Bowler | y = 190, x movable within lane |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/bowling-v0")
play("atari/bowling-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Release ball |
| `←` / `A` | Curve / step left |
| `→` / `D` | Curve / step right |
| `Esc` / close window | Quit |
