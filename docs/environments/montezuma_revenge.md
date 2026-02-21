# Montezuma's Revenge

> ALE name: `montezuma_revenge` · Game ID: `31`

Explore a multi-room underground pyramid, collecting keys and treasures, avoiding traps, and defeating enemies across 24 rooms.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for collecting keys, gems, swords, and torches, and for reaching new rooms. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when all lives are exhausted and the game-over screen is displayed, detected when the lives byte reaches zero and a screen-state byte equals `0x60`.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost by touching an enemy, falling too far, or touching a deadly environmental hazard.
