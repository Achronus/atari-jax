# Road Runner

> Game ID: `"atari/road_runner-v0"`

Run as fast as possible down the road while Wile E. Coyote chases you.
Eat birdseed for points; dodge trucks and the coyote.  Speed up or slow
down to outmanoeuvre hazards.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — beep (unused) |
| `2` | RIGHT — run faster |
| `3` | LEFT — slow down |
| `4` | UP — jump |
| `5` | DOWN — duck |

## Reward

Points are awarded for each birdseed pile eaten.

| Event | Points |
| --- | --- |
| Birdseed eaten | +100 |

## Episode End

The episode ends when all lives are lost.  A life is lost when the coyote
catches the player or a truck hits the player.

## Lives

The player starts with 5 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Road surface | y = 160 |
| Player | x = 30 (fixed), y = 148 (ground) |
| Coyote | approaches from behind (x starts at −30) |
| Trucks (3) | scroll left from x ≈ 170 |
| Birdseed (5) | scroll left; x ∈ [80, 200] initially |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/road_runner-v0")
play("atari/road_runner-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `→` / `D` | Run faster |
| `←` / `A` | Slow down |
| `↑` / `W` / `Space` | Jump |
| `Esc` / close window | Quit |
