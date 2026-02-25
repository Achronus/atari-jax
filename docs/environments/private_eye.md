# Private Eye

> Game ID: `"atari/private_eye-v0"`

A side-scrolling detective game.  Walk through city streets, collect clues
and arrest criminals.  Hook-shot to cross obstacles; use handcuffs to arrest
when you have enough clues.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — shoot hook |
| `2` | UP — jump / use hook |
| `3` | RIGHT |
| `4` | DOWN — duck / use handcuffs |
| `5` | LEFT |
| `6` | RIGHT + FIRE |
| `7` | LEFT + FIRE |
| `8` | UP + FIRE |

## Reward

Points are awarded for collecting clues and arresting criminals.

| Event | Points |
| --- | --- |
| Clue collected | +1 |
| Criminal arrested | +25 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a criminal
touches the player without the player attempting an arrest.

## Lives

The player starts with 5 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Ground | y = 160 |
| Player x range | x ∈ [10, 130] (world scrolls right) |
| Clues (8) | world x ∈ [200, 1500], spaced evenly |
| Criminals (3) | world x = 500, 900, 1300 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/private_eye-v0")
play("atari/private_eye-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Shoot hook |
| `↑` / `W` | Jump |
| `→` / `D` | Move right |
| `↓` / `S` | Use handcuffs |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
