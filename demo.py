import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the TreasureMazeGame environment
class TreasureMazeGame:
    def __init__(self):
        self.grid_size = 5
        self.reset()

    def reset(self):
        self.state = (0, 0)
        self.treasure = (4, 4)
        self.traps = [(1, 1), (2, 3), (3, 2)]
        return self.state

    def step(self, action):
        # Actions: 0 = up, 1 = right, 2 = down, 3 = left
        x, y = self.state
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.grid_size - 1:
            y += 1
        elif action == 2 and x < self.grid_size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1
        self.state = (x, y)

        if self.state in self.traps:
            return self.state, -10, True  # Fell into a trap
        if self.state == self.treasure:
            return self.state, 10, True  # Found the treasure
        return self.state, -0.1, False  # Each move has a small penalty

    def render(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        x, y = self.state
        grid[x][y] = 'R'  # Robot
        tx, ty = self.treasure
        grid[tx][ty] = 'T'  # Treasure
        for trap in self.traps:
            tx, ty = trap
            grid[tx][ty] = 'X'  # Traps
        print("\n".join(" ".join(row) for row in grid))
        print("---")

# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Define the DQN Agent
class DQNAgent:
    def __init__(
        self,
        state_size=2,    # (x, y) positions
        action_size=4,   # up, right, down, left
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=500,
        buffer_size=20000,
        min_buffer_size=1000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(capacity=buffer_size)
        self.min_buffer_size = min_buffer_size

    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
        else:
            return random.randint(0, self.action_size - 1)

    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.min_buffer_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Compute current Q values
        current_q = self.policy_net(state).gather(1, action)

        # Compute next Q values from target network
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + self.gamma * max_next_q * (1 - done)

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update the target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training the DQN Agent
def train_dqn():
    env = TreasureMazeGame()
    agent = DQNAgent()

    max_episodes = 1000
    max_steps = 100

    print("🚀 Initiating the robot's training journey...\n")

    rewards_history = deque(maxlen=50)  # To store rewards of the last 50 episodes

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps):
            # Normalize state
            normalized_state = np.array(state) / (env.grid_size - 1)

            action = agent.choose_action(normalized_state)
            next_state, reward, done = env.step(action)

            # Normalize next_state
            normalized_next_state = np.array(next_state) / (env.grid_size - 1)

            agent.push_memory(normalized_state, action, reward, normalized_next_state, done)
            agent.steps_done += 1

            # Perform a training step
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

        # Logging every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history)
            status = (
                "🌟 Outstanding!" if avg_reward > 8 else
                "👍 Progressing well." if avg_reward > 5 else
                "🔄 Learning continues."
            )
            print(f"Episode {episode:04}: Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f} | {status}")

    print("\n🎓 Training complete! The robot has honed its skills.\n")
    return agent

# Testing the DQN Agent
def test_agent(agent, max_test_steps=20):
    env = TreasureMazeGame()
    state = env.reset()
    done = False
    path = []
    print("🔍 Evaluating the robot's learned path:")
    env.render()

    for _ in range(max_test_steps):
        # Normalize state
        normalized_state = np.array(state) / (env.grid_size - 1)
        action = agent.choose_action(normalized_state, evaluate=True)
        path.append((state, action))
        next_state, reward, done = env.step(action)
        action_str = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}[action]
        print(f"👉 Moved {action_str} to {next_state} | Reward: {reward}")
        env.render()
        state = next_state

        if done:
            break

    if state == env.treasure:
        print("🎉 Success! The robot has found the treasure!")
    elif state in env.traps:
        print("⚠️ Oops! The robot fell into a trap. Time to train harder!")
    else:
        print("🔄 The robot couldn't find the treasure within the step limit.")

# Main Execution
if __name__ == "__main__":
    trained_agent = train_dqn()
    test_agent(trained_agent)

"""
🚀 Initiating the robot's training journey...

Episode 0050: Avg Reward: -10.81 | Epsilon: 1.00 | 🔄 Learning continues.
/.......:77: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:281.)
  state = torch.FloatTensor(state)
Episode 0100: Avg Reward: -7.98 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0150: Avg Reward: 7.34 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0200: Avg Reward: 7.42 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0250: Avg Reward: 7.76 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0300: Avg Reward: 5.72 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0350: Avg Reward: 3.81 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0400: Avg Reward: -4.04 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0450: Avg Reward: -11.16 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0500: Avg Reward: -11.77 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0550: Avg Reward: -7.41 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0600: Avg Reward: -1.54 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0650: Avg Reward: 4.65 | Epsilon: 0.10 | 🔄 Learning continues.
Episode 0700: Avg Reward: 6.00 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0750: Avg Reward: 6.40 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0800: Avg Reward: 7.08 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0850: Avg Reward: 5.66 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0900: Avg Reward: 7.61 | Epsilon: 0.10 | 👍 Progressing well.
Episode 0950: Avg Reward: 7.07 | Epsilon: 0.10 | 👍 Progressing well.
Episode 1000: Avg Reward: 4.81 | Epsilon: 0.10 | 🔄 Learning continues.

🎓 Training complete! The robot has honed its skills.

🔍 Evaluating the robot's learned path:
R . . . .
. X . . .
. . . X .
. . X . .
. . . . T
---
👉 Moved Down to (1, 0) | Reward: -0.1
. . . . .
R X . . .
. . . X .
. . X . .
. . . . T
---
👉 Moved Down to (2, 0) | Reward: -0.1
. . . . .
. X . . .
R . . X .
. . X . .
. . . . T
---
👉 Moved Right to (2, 1) | Reward: -0.1
. . . . .
. X . . .
. R . X .
. . X . .
. . . . T
---
👉 Moved Down to (3, 1) | Reward: -0.1
. . . . .
. X . . .
. . . X .
. R X . .
. . . . T
---
👉 Moved Down to (4, 1) | Reward: -0.1
. . . . .
. X . . .
. . . X .
. . X . .
. R . . T
---
👉 Moved Right to (4, 2) | Reward: -0.1
. . . . .
. X . . .
. . . X .
. . X . .
. . R . T
---
👉 Moved Right to (4, 3) | Reward: -0.1
. . . . .
. X . . .
. . . X .
. . X . .
. . . R T
---
👉 Moved Right to (4, 4) | Reward: 10
. . . . .
. X . . .
. . . X .
. . X . .
. . . . T
---
🎉 Success! The robot has found the treasure!
"""
