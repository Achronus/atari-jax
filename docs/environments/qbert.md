# Q*bert

> Game ID: `"atari/qbert-v0"`

Hop across an isometric pyramid of cubes, changing each one to the target colour
while avoiding Coily the snake and red balls.  Colour all 21 cubes to win.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(5)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | UP-RIGHT — jump up and right (row−1, col unchanged) |
| `2` | UP-LEFT — jump up and left (row−1, col−1) |
| `3` | DOWN-RIGHT — jump down and right (row+1, col+1) |
| `4` | DOWN-LEFT — jump down and left (row+1, col unchanged) |

## Reward

| Event | Reward |
| --- | --- |
| Land on an uncoloured cube | `+25` |
| Land on an already-coloured cube | `0` |
| Fall off pyramid or caught by enemy | `0` (life lost) |

## Episode End

The episode ends when either all 21 cubes are coloured (win) or all 3 lives are
lost.

## Lives

The player starts with 3 lives.  A life is lost by jumping off the pyramid
(invalid cell) or colliding with Coily or a red ball.  After each life loss
Qbert respawns at the apex; the cube colours are preserved.

## Pyramid Layout

The pyramid has 6 rows (row 0 = apex, row 5 = base).  Valid cells satisfy
`0 ≤ col ≤ row`.  Total valid cells: 1+2+3+4+5+6 = 21.

| Direction | Row change | Col change |
| --- | --- | --- |
| UP-RIGHT | −1 | 0 |
| UP-LEFT | −1 | −1 |
| DOWN-RIGHT | +1 | +1 |
| DOWN-LEFT | +1 | 0 |

Jumping to a cell where `col < 0`, `col > row`, or `row > 5` counts as falling
off the pyramid and costs a life.

## Enemies

| Enemy | Behaviour |
| --- | --- |
| Coily (snake) | Spawns at apex, chases Qbert downward; deactivates when it falls off the base |
| Red ball | Spawns at apex, always moves down-right; deactivates when it falls off the base |

Enemies spawn every 60 emulated frames, alternating randomly between Coily and a
red ball.

## Interactive Play

```python
from atarax.utils.render import play

play("atari/qbert-v0")                          # 480×630 window, ALE-accurate speed
play("atari/qbert-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `→` | Jump up-right |
| `←` | Jump up-left |
| `↓` | Jump down-left |
| `Esc` / close window | Quit |
