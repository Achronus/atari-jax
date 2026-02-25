# Space Invaders

> Game ID: `"atari/space_invaders-v0"`

Defend Earth by shooting descending alien invaders before they reach the ground.
The alien formation accelerates as invaders are destroyed.

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
| `2` | RIGHT — move cannon right |
| `3` | LEFT — move cannon left |
| `4` | RIGHT + FIRE |
| `5` | LEFT + FIRE |

## Reward

Points are awarded for each alien destroyed.  The reward on each `step` call is
the sum earned across the four emulated frames in the frame-skip window.

| Alien row (top → bottom) | Points |
| --- | --- |
| Row 0 (top) | 30 |
| Rows 1–2 | 20 |
| Rows 3–4 (bottom) | 10 |

## Episode End

The episode ends when any of the following occurs:

| Condition | Description |
| --- | --- |
| All aliens destroyed | Player clears the formation |
| Formation reaches ground | Bottom of alien grid reaches y = 185 |
| Lives exhausted | All 3 lives are lost |

## Lives

The player starts with 3 lives.  A life is lost when an alien bullet hits the
cannon.

## Alien Movement

The formation moves horizontally, reversing direction and dropping 8 pixels
whenever it reaches the play-area boundary.  Movement speed scales with
remaining aliens: at full count (55 aliens) the formation steps every 6
emulated frames; as invaders are killed the interval shortens proportionally,
reaching 1 frame per step with a single alien remaining.

## Screen Geometry

| Element | Position |
| --- | --- |
| Playfield | x ∈ [8, 152), y ∈ [30, 185) |
| Cannon | y ∈ [177, 185), w=13 px; x ∈ [8, 139) |
| Alien grid | 5 rows × 11 cols, 8×8 px, 10 px col step, 16 px row step |
| Initial formation | top-left at x=26, y=50 |
| Player bullet | 1×4 px, moves up 4 px/frame |
| Alien bullet | 1×4 px, moves down 2 px/frame |
| Ground line | y = 185 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/space_invaders-v0")                 # 480×630 window, ALE-accurate speed
play("atari/space_invaders-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `→` / `D` | Move cannon right |
| `←` / `A` | Move cannon left |
| `Space` | Fire |
| `Esc` / close window | Quit |
