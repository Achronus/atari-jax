# Asterix

> Game ID: `"atari/asterix-v0"`

Guide Asterix through a side-scrolling world collecting magic potions while
avoiding enemies.  Items scroll from right to left across 8 fixed lanes.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(5)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | UP — move one lane up |
| `2` | DOWN — move one lane down |
| `3` | LEFT |
| `4` | RIGHT |

## Reward

Collecting a magic potion earns points.

| Event | Points |
| --- | --- |
| Magic potion collected | +50 |

Touching an enemy costs a life; no points deducted.

## Episode End

The episode ends when all lives are lost.  A life is lost each time the
player touches an enemy item.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Lanes | 8 horizontal lanes y ∈ {30, 52, 74, 96, 118, 140, 162, 184} |
| Player | x = 40 (fixed), starts in lane 3 |
| Items | spawn at x = 155, scroll left |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/asterix-v0")
play("atari/asterix-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | Move lane up |
| `↓` / `S` | Move lane down |
| `Esc` / close window | Quit |
