# Seaquest

> Game ID: `"atari/seaquest-v0"`

Pilot a submarine underwater, rescuing divers and shooting enemy submarines and sharks. Surface periodically to replenish oxygen before it runs out.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP (surface / ascend) |
| `3` | RIGHT |
| `4` | DOWN (descend) |
| `5` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Enemy sub shot | `+20` |
| Shark shot | `+20` |
| Diver rescued (per diver carried when surfacing) | `+50` |

## Episode End

The episode ends when all lives are lost. A life is lost by colliding with an enemy or by allowing the oxygen supply to run out without surfacing.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Surface line | y = 20 |
| Bottom boundary | y = 190 |
| Oxygen bar | top strip, rows 0–9 |
| Player submarine | clipped to x ∈ [0, 152], y ∈ [20, 190] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/seaquest-v0")
play("atari/seaquest-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP (ascend / surface) |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN (descend) |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
