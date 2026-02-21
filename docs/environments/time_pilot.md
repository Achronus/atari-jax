# Time Pilot

> ALE name: `time_pilot` · Game ID: `49`

Fly a fighter plane through five historical eras, from 1910 biplanes to 2001 UFOs, destroying enemies and rescuing stranded soldiers.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying enemy aircraft and for rescuing soldiers; scores are multiples of 100. Score is stored as packed BCD across two non-sequential RAM bytes.

## Episode End

The episode ends when a game-over flag byte becomes non-zero.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost each time the player's plane is shot down.
