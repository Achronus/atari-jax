# Solaris

> ALE name: `solaris` · Game ID: `45`

Pilot a starship across a galactic map, entering star systems to fight off enemy fleets and liberate planets from alien occupation.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying enemy ships in combat. Score is stored as packed BCD across three RAM bytes, with values being multiples of 10.

## Episode End

The episode ends when the lives RAM byte reaches zero.

## Lives

Lives are stored in the low nibble of a RAM byte. A life is lost each time the player's starship is destroyed in battle.
