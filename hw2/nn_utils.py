import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import gym
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque

from utils import BasePolicy, EMPTY, CROSSES_TURN, NOUGHTS_TURN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, env, hidden_dim: int = 256):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(env.n_rows, 2 * hidden_dim, env.n_cols).to(device)
        self.l1 = nn.Linear(2 * hidden_dim, hidden_dim).to(device)
        self.l2 = nn.Linear(hidden_dim, env.n_rows * env.n_cols).to(device)       
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
    
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size: int = 64):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNAgent(BasePolicy):

    def reset(self):
        self.action = None 
        self.reward = None
        self.board = None
        self.new_board = None

    def __init__(self, turn, model: nn.Module, lr: float = 1e-5, memory_capacity: int = 30000):
        super().__init__(turn)
        self.reset()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.memory = ReplayMemory(capacity=memory_capacity)


    def select_action(self, state, eps: float = 0.1):
        is_action_int = True
        _, _, _, board = state
        if random.random() < eps:
            action_int = random.choice(np.nonzero(board.flatten() == 0)[0])
            return action_int, is_action_int

        self.model.eval()
        board = np.stack(
            [
                (board == CROSSES_TURN).astype(float),
                (board == NOUGHTS_TURN).astype(float),
                (board == EMPTY).astype(float)
            ]
        )
        board = torch.FloatTensor(board).unsqueeze(0).to(device)
        action_int = self.model(board).detach().max(1).indices.item()
        return action_int, is_action_int
        
    def update(self, new_state, action, reward, done):
        _, _, _, new_board = new_state
        new_board = np.stack(
            [
                (new_board == CROSSES_TURN).astype(float),
                (new_board == NOUGHTS_TURN).astype(float),
                (new_board == EMPTY).astype(float)
            ]
        )
        if self.board is not None:
            self.memory.store((self.board, new_board, self.action, reward, done))
        self.board = new_board
        self.action = action
    
    def train_batch(self, batch_size=256, gamma=0.9):    
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        board, next_board, actions, rewards, dones = list(zip(*transitions))
        
        states = np.array(board)
        next_states = np.array(next_board)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        batch_board = torch.FloatTensor(board).to(device)
        batch_next_board = torch.FloatTensor(next_board).to(device)
        batch_actions = torch.LongTensor(actions).reshape(-1, 1).to(device)
        batch_reward = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        batch_dones = torch.BoolTensor(rewards).reshape(-1, 1).to(device)

        self.model.train()
        Q = self.model(batch_board).gather(1, batch_actions)

        with torch.no_grad():
            Qnext = self.model(batch_next_board).max(dim=1).values.reshape(-1, 1)
            Qnext[batch_dones] = 0
            Qnext = batch_reward + gamma * Qnext

        loss = F.l1_loss(Q, Qnext)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    
def dqn_learning_train_epoch(env, agents, args):
    env.reset()
    for key in agents:
        agents[key].model.eval()
        agents[key].reset()
        
    state = env.getState()

    done = False
    while not done:
        _, _, turn, _ = state
        current = agents[turn]
        action_int, _ = current.select_action(state, args['eps'])
        current.update(state, action_int, 0, done)
        state, reward, done, _ = env.step(env.action_from_int(action_int))

    if reward == -10:
        current.update(state, action_int, reward, done)
    else:
        agents[CROSSES_TURN].update(state, action_int, reward, done)
        agents[NOUGHTS_TURN].update(state, action_int, -reward, done)
    
    agents[CROSSES_TURN].train_batch(args['batch_size'], args['gamma'])
    agents[NOUGHTS_TURN].train_batch(args['batch_size'], args['gamma'])
    return agents


class DDQNAgent(DQNAgent):

    def __init__(self, turn, model: nn.Module, lr: float = 1e-5, memory_capacity: int = 30000):
        super().__init__(turn, model, lr, memory_capacity)
        self.steps = 0
        self.target_model = copy.deepcopy(self.model) 
    
    def __train_batch(self, batch_size=256, gamma=0.9):    
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        states, next_states, actions, rewards, dones = list(zip(*transitions))
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        batch_state = torch.FloatTensor(states).to(device)
        batch_next_state = torch.FloatTensor(next_states).to(device)
        batch_actions = torch.LongTensor(actions).reshape(-1, 1).to(device)
        batch_reward = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        batch_dones = torch.BoolTensor(rewards).reshape(-1, 1).to(device)

        self.model.train()
        Q = self.model(batch_state).gather(1, batch_actions)

        with torch.no_grad():
            Qnext = self.target_model(batch_next_state).max(dim=1).values.reshape(-1, 1)
            Qnext[batch_dones] = 0
            Qnext = batch_reward + gamma * Qnext

        loss = F.l1_loss(Q, Qnext)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train_batch(
        self, batch_size=256, gamma=0.9,
        steps_per_update=4, 
        steps_per_target_update=4*100
    ):
        if self.steps % steps_per_update == 0:
            self.__train_batch(batch_size, gamma)
        if self.steps % steps_per_target_update == 0:
            self.update_target_model()
        self.steps += 1