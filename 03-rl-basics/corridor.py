import random

N = 6  # squares 0..5, the goal is square 5


def step(state, action):
    """action 0 = left, 1 = right. Returns (new_state, reward, done)."""
    if action == 1:
        state = min(state + 1, N - 1)
    else:
        state = max(state - 1, 0)
    if state == N - 1:
        return state, 1.0, True
    return state, 0.0, False


def run_episode(max_steps=200):
    state = 0
    for t in range(1, max_steps + 1):
        action = random.choice([0, 1])  # a random "policy"
        state, reward, done = step(state, action)
        if done:
            return t
    return max_steps


lengths = [run_episode() for _ in range(1000)]
print("average steps to reach the goal (random agent):", sum(lengths) / len(lengths))
