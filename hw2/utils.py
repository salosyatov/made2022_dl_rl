import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict, deque

EMPTY = 0
CROSSES_TURN = 1
NOUGHTS_TURN = -1

class TicTacToe(gym.Env):
    def __init__(self, n_rows, n_cols, n_win, clone=None):
        if clone is not None:
            self.n_rows = clone.n_rows
            self.n_cols = clone.n_cols
            self.n_win = clone.n_win
            self.board = copy.deepcopy(clone.board)
            self.curTurn = clone.curTurn
            self.emptySpaces = None
            self.boardHash = None
        else:
            self.n_rows = n_rows
            self.n_cols = n_cols
            self.n_win = n_win

            self.reset()

    def getEmptySpaces(self):
        if self.emptySpaces is None:
            res = np.where(self.board == 0)
            self.emptySpaces = np.array([(i, j) for i, j in zip(res[0], res[1])])
        return self.emptySpaces

    def makeMove(self, player, i, j):
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def getHash(self):
        if self.boardHash is None:
            self.boardHash = "".join(
                [f"{x + 1}"
                 for x in self.board.reshape(self.n_rows * self.n_cols)]
            )
        return self.boardHash

    def isTerminal(self):
        # проверим, не закончилась ли игра
        cur_marks, cur_p = np.where(self.board == self.curTurn), self.curTurn
        for i, j in zip(cur_marks[0], cur_marks[1]):
            win = False
            if i <= self.n_rows - self.n_win:
                if np.all(self.board[i:i + self.n_win, j] == cur_p):
                    win = True
            if not win:
                if j <= self.n_cols - self.n_win:
                    if np.all(self.board[i, j:j + self.n_win] == cur_p):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j <= self.n_cols - self.n_win:
                    if np.all(np.array([self.board[i + k, j + k] == cur_p
                                        for k in range(self.n_win)])):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j >= self.n_win - 1:
                    if np.all(np.array([self.board[i + k, j - k] == cur_p
                                        for k in range(self.n_win)])):
                        win = True
            if win:
                self.gameOver = True
                return self.curTurn

        if len(self.getEmptySpaces()) == 0:
            self.gameOver = True
            return 0

        self.gameOver = False
        return None

    def printBoard(self):
        for i in range(0, self.n_rows):
            print("----" * (self.n_cols) + "-")
            out = "| "
            for j in range(0, self.n_cols):
                if self.board[i, j] == 1:
                    token = "x"
                if self.board[i, j] == -1:
                    token = "o"
                if self.board[i, j] == 0:
                    token = " "
                out += token + " | "
            print(out)
        print("----" * (self.n_cols) + "-")

    def getState(self):
        return (self.getHash(), self.getEmptySpaces(), self.curTurn, self.board)

    def action_from_int(self, action_int):
        return (int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]
    
    @staticmethod
    def next_turn(turn):
        return -turn

    def step(self, action):
        if self.board[action[0], action[1]] != 0:
            return self.getState(), -10, True, {}
        self.makeMove(self.curTurn, action[0], action[1])
        reward = self.isTerminal()
        self.curTurn = self.next_turn(self.curTurn)
        return self.getState(), 0 if reward is None else reward, reward is not None, {}

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = 1
        
        
class BasePolicy:
    def __init__(self, turn):
        self.turn = turn
    
    def check_win(self, reward):
        if reward == CROSSES_TURN:
            return 1 if self.turn == CROSSES_TURN else -1
        if reward == NOUGHTS_TURN:
            return 1 if self.turn == NOUGHTS_TURN else -1
        return 0 # draw
    
    def select_action_idx(self, state, eps):
        raise NotImplementedError()
    
    def select_action(self, state, eps):
        is_action_int = False
        _, allowed_actions, _, _ = state
        action_idx = self.select_action_idx(state, eps)
        return allowed_actions[action_idx], is_action_int

    
class RandomPolicy(BasePolicy):
    
    def select_action_idx(self, state, eps):
        _, allowed_actions, _, _ = state
        action_idx = np.random.randint(len(allowed_actions))
        return action_idx

    
class EpsGreedyPolicy(BasePolicy):
    def __init__(self, turn):
        super().__init__(turn)
        self.Q = {}
    
    def select_action_idx(self, state, eps):
        board_hash, allowed_actions, _, _ = state
        
        if random.random() >= eps and board_hash in self.Q:
            return np.argmax(self.Q[board_hash])

        if board_hash not in self.Q:
            self.Q[board_hash] = np.zeros(len(allowed_actions))

        return np.random.randint(len(allowed_actions))
    
def run_eval_episode(env, policies, eps, ):
    
    env.reset()
    state = env.getState()
    _, _, turn, _ = state

    done = False
    while not done:
        action, is_action_int = policies[turn].select_action(state, eps)
        action = action if not is_action_int else env.action_from_int(action)
        state, reward, done, _ = env.step(action)
        _, _, turn, _ = state

    return reward

def eval_policy(env, pi_eval, pi_other, episodes, verbose=False):
    """
    eval pi_eval vs pi_other,
    returns win rate, lose rate, draw rate
    """
    policies = dict()
    policies[CROSSES_TURN] = pi_eval  if pi_eval.turn == CROSSES_TURN  else pi_other
    policies[NOUGHTS_TURN] = pi_other if pi_other.turn == NOUGHTS_TURN else pi_eval
    assert policies[CROSSES_TURN].turn == CROSSES_TURN
    assert policies[NOUGHTS_TURN].turn == NOUGHTS_TURN
    
    wins = 0
    loses = 0
    draws = 0

    for _ in tqdm(range(episodes), disable = not verbose):
        reward = run_eval_episode(env, policies, eps=0)
        is_win = pi_eval.check_win(reward)
        wins += int(is_win == 1)
        loses += int(is_win == -1)
        draws += int(is_win == 0)

    return wins / episodes, loses / episodes, draws / episodes

def plot_history(history, player, prefix='', fontsize: int = 16, figsize: tuple = (14, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'{prefix}Average rates during training: {player}', fontsize=fontsize)
    plt.xlabel('Step', fontsize=fontsize)
    plt.ylabel('Average rate', fontsize=fontsize)
    history = history[player]
    x = [s for s, _, _, _ in history]
    wr = [s for _, s, _, _ in history]
    lr = [s for _, _, s, _ in history]
    dr = [s for _, _, _, s in history]
    plt.plot(x, wr, label="win rate", color="green")
    plt.plot(x, lr, label="lose rate", color="yellow")
    plt.plot(x, dr, label="draw rate", color="red")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    
    
def q_learning_train_epoch(env, policies, args):

    env.reset()
    state = env.getState()
    board_hash, allowed_actions, turn, _ = state
    prev_board_hash, prev_action_idx = None, None

    done = False
    while not done:
        
        # update curr turn policy
        pi = policies[turn]
        action_idx = pi.select_action_idx(state, args['eps'])
        action = allowed_actions[action_idx]
        state, reward, done, _ = env.step(action)
        if reward == turn:
            pi.Q[board_hash][action_idx] = abs(reward)
        policies[turn] = pi

        # update opposite policy
        next_board_hash, next_allowed_actions, next_turn, _ = state
        pi = policies[next_turn]
        if prev_board_hash is not None:
            gamma_term = args['gamma'] * np.max(pi.Q[next_board_hash]) if next_board_hash in pi.Q else 0
            pi.Q[prev_board_hash][prev_action_idx] += args['alpha'] * (
                -reward + gamma_term - pi.Q[prev_board_hash][prev_action_idx]
            )
        policies[next_turn] = pi

        prev_board_hash, prev_action_idx = board_hash, action_idx
        board_hash, allowed_actions, turn, _ = state

    return policies

def run_learning(env, pi_crosses, pi_noughts, run_train_epoch, args,
                 episodes = 10000, eps_init = 0.9, eps_final = 0.01):
    
    eval_every = int(episodes * 0.05)
    eval_episodes = int(episodes * 0.05)
    eps_decay = int(episodes * 0.8)
    
    policies = dict()
    policies[CROSSES_TURN] = pi_crosses
    policies[NOUGHTS_TURN] = pi_noughts
    assert pi_crosses.turn == CROSSES_TURN
    assert pi_noughts.turn == NOUGHTS_TURN
    
    hist = defaultdict(list)
    for epoch in tqdm(range(episodes)):
        
        args['eps'] = eps_init + (eps_final - eps_init) * epoch / eps_decay
        policies = run_train_epoch(env, policies, args)

        if (epoch + 1) % eval_every == 0:
            step = epoch + 1
            wr, lr, dr = eval_policy(env, policies[CROSSES_TURN], RandomPolicy(NOUGHTS_TURN), eval_episodes)
            hist['crosses'].append((step, wr, lr, dr))
            wr, lr, dr = eval_policy(env, policies[NOUGHTS_TURN], RandomPolicy(CROSSES_TURN), eval_episodes)
            hist['noughts'].append((step, wr, lr, dr))

    return hist