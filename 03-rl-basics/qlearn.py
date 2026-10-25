import random

N = 6  # squares 0..5, the goal is square 5
ACTIONS = [0, 1]  # 0 = left, 1 = right


def step(state, action):
    if action == 1:
        state = min(state + 1, N - 1)
    else:
        state = max(state - 1, 0)
    if state == N - 1:
        return state, 1.0, True
    return state, 0.0, False


Q = [[0.0, 0.0] for _ in range(N)]  # Q[state][action]
alpha = 0.5    # learning rate
gamma = 0.9    # how much future reward counts
epsilon = 0.2  # chance to explore


def best_action(state):
    q = Q[state]
    if q[0] == q[1]:
        return random.choice(ACTIONS)
    return 0 if q[0] > q[1] else 1


for episode in range(200):
    state = 0
    for t in range(200):
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = best_action(state)
        new_state, reward, done = step(state, action)
        target = reward + (0.0 if done else gamma * max(Q[new_state]))
        Q[state][action] += alpha * (target - Q[state][action])
        state = new_state
        if done:
            break

print("Q-table, [left, right] per square:")
for s in range(N):
    print(s, [round(v, 3) for v in Q[s]])

# test the learned policy with no exploring
state = 0
steps = 0
done = False
while not done and steps < 100:
    state, _, done = step(state, best_action(state))
    steps += 1
print("steps with the learned policy:", steps)
