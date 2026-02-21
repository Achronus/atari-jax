# Tutankham

> ALE name: `tutankham` · Game ID: `50`

Navigate through the labyrinthine tomb of Tutankham, shooting enemies with a limited-shot laser and collecting treasures across four levels.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting enemies and collecting treasures in the tomb. Score is stored as packed BCD across two non-sequential RAM bytes.

## Episode End

The episode ends when the lives byte reaches zero, provided the game-loaded-but-not-started guard value is not active, preventing false terminals during initialisation.

## Lives

Lives are stored in the lower 2 bits of a RAM byte (values 0–3). A life is lost by being caught by an enemy or touching a hazard in the tomb.
