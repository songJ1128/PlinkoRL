import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


num_positions = 10
gamma = 0.99  #discount
epsilon_start = 1.0  # Start val
epsilon_end = 0.1  # final val
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
target_update_freq = 10
memory_size = 10000
num_episodes = 500
max_steps = 10
bucket_multipliers = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9, 1.0,1.1,1.2,1.3,1.4, 1.5,1.6,1.7,1.8,1.9, 2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9, 3.0])  # Example bucket multipliers

class PlinkoEnv:
    def __init__(self):
        self.num_positions = num_positions
        self.bucket_multipliers = bucket_multipliers
        self.board = self._create_board()

    def _create_board(self):

        board = np.zeros((num_positions, num_positions))

        board[1, 4] = 1
        board[2, 2] = 1
        board[3, 7] = 1
        return board

    def reset(self):
        return np.zeros(self.num_positions)

    def step(self, action):



        position = action
        for row in range(self.board.shape[0]):
            if self.board[row, position] == 1:
                position = max(0, min(position + random.choice([-1, 1]), self.num_positions - 1))

        bucket = position
        multiplier = self.bucket_multipliers[bucket % len(self.bucket_multipliers)]
        reward = multiplier #reward multiplier
        next_state = np.zeros(self.num_positions)
        next_state[action] = 1
        return next_state, reward, bucket

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_positions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_positions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train():
    env = PlinkoEnv()
    q_network = QNetwork()
    target_network = QNetwork()
    target_network.load_state_dict(q_network.state_dict())  # Sync target network
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    memory = ReplayMemory(memory_size)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)


            if random.random() < epsilon:
                action = random.randint(0, num_positions - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()


            next_state, reward, _ = env.step(action)
            total_reward += reward


            memory.push(state, action, reward, next_state)

            state = next_state


            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

                batch_state_tensor = torch.FloatTensor(batch_state)
                batch_action_tensor = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward_tensor = torch.FloatTensor(batch_reward)
                batch_next_state_tensor = torch.FloatTensor(batch_next_state)

                #calculate q val
                current_q_values = q_network(batch_state_tensor).gather(1, batch_action_tensor)


                with torch.no_grad():
                    max_next_q_values = target_network(batch_next_state_tensor).max(1)[0]
                    target_q_values = batch_reward_tensor + gamma * max_next_q_values

                loss = criterion(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if episode % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())


        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        print(f"iteration {episode + 1}, total reward: {total_reward}")


train()
