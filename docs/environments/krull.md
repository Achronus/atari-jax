# Krull

> ALE name: `krull` · Game ID: `29`

Fight through multiple scenes based on the film Krull, wielding the magical Glaive weapon to defeat enemies and rescue Princess Lyssa.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for defeating enemies across the various scenes. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when all three game-over conditions are simultaneously met: the lives byte reaches zero, a status byte equals `0x03`, and a second status byte equals `0x80`.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost each time the player character is defeated by an enemy.
