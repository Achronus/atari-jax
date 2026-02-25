# Tennis

> Game ID: `"atari/tennis-v0"`

One-on-one tennis against a CPU opponent. Win points by hitting the ball past the opponent. Points follow standard tennis scoring; first to win 6 games wins the match.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (swing racket) |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

The reward is `+1` when the player wins a point (ball passes the CPU baseline) and `-1` when the CPU wins a point (ball passes the player baseline). Points follow tennis scoring: 0, 15, 30, 40, game. The `score` field tracks the number of games won by the player.

## Episode End

The episode ends when either player reaches 6 games won.

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Court left | x = 10 |
| Court right | x = 150 |
| Court top | y = 30 |
| Court bottom | y = 185 |
| Net | y = 107 |
| Player start | y = 160 |
| CPU start | y = 55 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/tennis-v0")
play("atari/tennis-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (swing racket) |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
