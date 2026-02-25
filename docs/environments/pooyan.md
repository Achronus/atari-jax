# Pooyan

> Game ID: `"atari/pooyan-v0"`

A mama pig defends her piglets from wolves descending on balloons.  The
player's basket moves up and down on the left side; shoot arrows to pop
balloons and hit wolves before they reach the basket.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(5)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — shoot arrow |
| `2` | UP |
| `3` | DOWN |
| `4` | FIRE + UP |

## Reward

Points are awarded for each wolf balloon popped.

| Event | Points |
| --- | --- |
| Balloon popped (wolf falls) | +110 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a wolf
reaches the basket at the left edge.

## Lives

The player starts with 5 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Basket (player) | x = 10, y ∈ [20, 175] |
| Basket rope | x ∈ [10, 14], y ∈ [20, 190] |
| Wolves on balloons | approach from x = 150, fixed y positions |
| Arrow | flies rightward from basket |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/pooyan-v0")
play("atari/pooyan-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Shoot arrow |
| `↑` / `W` | Move basket up |
| `↓` / `S` | Move basket down |
| `Esc` / close window | Quit |
