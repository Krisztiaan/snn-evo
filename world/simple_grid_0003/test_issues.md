# Test Suite Issues

After reviewing the test suite, here are the issues I've identified:

## 1. Movement Direction Issue
The test assumes:
- Action 0 = up (decreases y)
- Action 1 = right (increases x)
- Action 2 = down (increases y)  
- Action 3 = left (decreases x)

But the implementation has movement deltas that may not match this expectation.

## 2. Agent Position Type Inconsistency
- Tests expect `state.agent_pos` to be a tuple of ints: `(10, 10)`
- But implementation returns JAX arrays: `(Array(10), Array(10))`
- This causes assertion failures when comparing positions

## 3. Reward Collection Test Logic
The test `test_reward_collection` places agent one step below reward and moves up, but:
- It assumes the agent will end up exactly at the reward position
- Doesn't account for potential edge wrapping
- Doesn't verify that the reward was actually at that position before collection

## 4. Gradient Test Assumptions
- `test_observation_gradient_when_far` uses Manhattan distance which doesn't match the Euclidean distance used by the world
- The threshold values (0.1, 0.5, 0.8) are arbitrary and may not match actual gradient calculations

## 5. Missing Edge Cases
- No test for collecting multiple rewards in one step (if agent is on multiple rewards)
- No test for reward positions being unique after respawn
- No test for gradient when all rewards are collected

## 6. Test Design Issues
- `test_no_collection_of_already_collected` places agent ON a collected reward then steps, but agent will move away
- `test_can_collect_rewards_until_done` is overly complex for what it's testing

## 7. API Contract Issues
- Tests don't verify that `result.reward` is exactly the count of collected rewards
- No test for what happens when multiple rewards are at the same position
- No test that collected rewards always respawn