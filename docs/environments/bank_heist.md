# Bank Heist

> Game ID: `"atari/bank_heist-v0"`

Drive a getaway car through a scrolling city robbing banks and evading
police.  Drop dynamite bombs to destroy pursuing police cars.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(10)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — drop dynamite |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | UP + FIRE |
| `7` | RIGHT + FIRE |
| `8` | DOWN + FIRE |
| `9` | LEFT + FIRE |

## Reward

Points are awarded for robbing banks and destroying police cars.

| Event | Points |
| --- | --- |
| Bank robbed | +50 |
| Police car destroyed | +25 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a police
car catches the getaway car.

## Lives

The player starts with 4 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Banks | 8 banks across the city grid |
| Player start | x = 80, y = 170 |
| Playfield | x ∈ [5, 155], y ∈ [20, 195] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/bank_heist-v0")
play("atari/bank_heist-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Drop dynamite |
| `↑` / `W` | Drive up |
| `→` / `D` | Drive right |
| `↓` / `S` | Drive down |
| `←` / `A` | Drive left |
| `Esc` / close window | Quit |
