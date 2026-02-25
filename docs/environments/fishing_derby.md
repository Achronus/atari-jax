# Fishing Derby

> Game ID: `"atari/fishing_derby-v0"`

Compete against a CPU angler to catch the most valuable fish before either player reaches 99 points. A shark patrols the middle depths and can snap the line, forcing a recast.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (reel in / cast) |
| `2` | UP (raise line) |
| `3` | RIGHT (move along dock) |
| `4` | DOWN (lower line) |
| `5` | LEFT (move along dock) |

## Reward

The reward is the change in score differential (player score minus CPU score) per step.

| Fish depth | Points awarded |
| --- | --- |
| Shallow (y ≈ 90) | +2 |
| Mid (y ≈ 120) | +4 |
| Deep (y ≈ 150) | +6 |

When the CPU catches a fish, the step reward is reduced by that fish's point value.

## Episode End

The episode ends when either the player or the CPU reaches 99 points. The episode also ends after the time limit (9600 emulated frames, approximately 2400 agent steps).

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Dock | y = 50 |
| Water surface | y = 70 |
| Water bottom | y = 185 |
| Player dock x | x = 20 |
| CPU dock x | x = 140 |
| Shark patrol y | y = 130 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/fishing_derby-v0")
play("atari/fishing_derby-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (reel in / cast) |
| `↑` / `W` | UP (raise line) |
| `↓` / `S` | DOWN (lower line) |
| `Esc` / close window | Quit |
