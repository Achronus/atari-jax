# Ice Hockey

> Game ID: `"atari/ice_hockey-v0"`

Two-player ice hockey game. The player controls a skater and tries to shoot the puck into the opponent's net. First to score 6 goals wins, or the higher score after the time limit.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (shoot puck) |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Goal scored | +1 |
| Goal conceded | -1 |

## Episode End

The episode ends when either team reaches 6 goals. The episode also ends after the time limit (9600 emulated frames, approximately 2400 agent steps).

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Rink | x ∈ [10, 150], y ∈ [30, 185] |
| Left net (player shoots here) | x ∈ [6, 10] |
| Right net (CPU shoots here) | x ∈ [150, 154] |
| Goal mouth y range | y ∈ [97, 117] |
| Centre line | x = 80 |
| Puck face-off | x = 75, y = 107 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/ice_hockey-v0")
play("atari/ice_hockey-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (shoot puck) |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
