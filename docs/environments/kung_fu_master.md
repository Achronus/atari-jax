# Kung-Fu Master

> Game ID: `"atari/kung_fu_master-v0"`

Fight through five floors of a pagoda, punching and kicking enemies to
rescue Princess Silvia from the Devil King.  Reach the right edge of each
floor to advance.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — punch |
| `2` | UP — jump |
| `3` | RIGHT |
| `4` | DOWN — crouch / kick low |
| `5` | LEFT |
| `6` | RIGHT + FIRE |
| `7` | LEFT + FIRE |
| `8` | DOWN + FIRE — low kick |

## Reward

Points are awarded for defeating enemies and clearing floors.

| Event | Points |
| --- | --- |
| Enemy defeated | +100 |
| Floor cleared (reach right edge) | +500 |

## Episode End

The episode ends when all lives are lost or all 5 floors are cleared.  A
life is lost when an enemy touches the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Floor surface | y = 170 |
| Player x range | x ∈ [5, 155] |
| Player start | x = 20 |
| Enemies (5) | spawn from left or right edge at y = 170 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/kung_fu_master-v0")
play("atari/kung_fu_master-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Punch |
| `↑` / `W` | Jump |
| `→` / `D` | Move right |
| `↓` / `S` | Crouch |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
