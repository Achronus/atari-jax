# Yar's Revenge

> ALE name: `yars_revenge` · Game ID: `55`

Control a housefly-like creature that must eat or shoot through a rotating shield to expose the Qotile enemy and destroy it with a supercharged cannon.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for eating shield cells, shooting shield cells with the Zorlon Cannon, and for destroying the Qotile. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the upper nibble of the lives RAM byte reaches zero.

## Lives

Lives are stored in the upper nibble of a RAM byte as a 0-based count. A life is lost each time the player's Yar is hit by the Qotile's missile or touches the Qotile directly.
