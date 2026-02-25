# Chopper Command

> Game ID: `"atari/chopper_command-v0"`

Pilot a helicopter defending a convoy against enemy planes and tanks. Enemies approach from both sides; shoot them before they reach the trucks.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(7)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | UP + FIRE |

## Reward

| Event | Reward |
| --- | --- |
| Enemy jet shot | +100 |
| Enemy tank shot | +200 |
| Convoy truck destroyed | -100 |

## Episode End

The episode ends when the helicopter is hit by an enemy jet and all lives are lost.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Sky (jet flight altitude) | y = 50 |
| Ground strip | y >= 170 |
| Truck convoy | y = 165 |
| Chopper x range | x ∈ [8, 152] |
| Chopper y range | y ∈ [30, 160] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/chopper_command-v0")
play("atari/chopper_command-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
