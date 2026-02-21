# James Bond

> ALE name: `jamesbond` · Game ID: `27`

Play as secret agent 007 in a multi-mode vehicle that can function as a car, boat, and helicopter, completing missions by defeating enemies.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying enemy vehicles and installations. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the lives nibble reaches zero and the death-screen byte is set to its sentinel value (`0x68`), confirming the game-over screen is displayed.

## Lives

Lives are stored as a zero-based count in the low nibble of a RAM byte (displayed lives = nibble + 1). A life is lost each time the player's vehicle is destroyed.
