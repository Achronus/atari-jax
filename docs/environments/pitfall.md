# Pitfall!

> ALE name: `pitfall` · Game ID: `35`

Guide Pitfall Harry through a jungle within 20 minutes, swinging on vines, jumping pits, and collecting 32 treasures while avoiding hazards.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the change in score on each step. Points are earned by collecting treasures; touching hazards deducts points. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the lives nibble (upper nibble of a RAM byte) reaches zero and the game-over logo timer is non-zero, indicating the final death sequence is playing.

## Lives

The player starts with 3 lives encoded in the upper nibble of a RAM byte (nibble values `0xA` = 3 lives, `0x8` = 2 lives, otherwise = 1 life). A life is lost by touching a crocodile, scorpion, snake, fire, or falling into a tar pit.
