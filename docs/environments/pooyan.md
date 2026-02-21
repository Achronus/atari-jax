# Pooyan

> ALE name: `pooyan` · Game ID: `37`

Ride a cable car up and down a cliff, shooting wolves that descend on balloons to protect your family of piglets below.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting wolves and bursting their balloons. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when all lives are exhausted and the game-over screen is shown, detected when the lives byte reaches zero and a terminal screen byte equals `0x05`.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost each time a wolf reaches the bottom of the cliff and kidnaps a piglet.
