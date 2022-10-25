import random
from joblib import Parallel, delayed

from tqdm import trange
import numpy as np
import gym

from blackjack_utils import get_index_from_card_count


EPSILON_INIT = 0.1
EPSILON_END = 0.001

CONTROL_INTERVAL = 5000
N_iters = 1_000_000

def get_average_rewards(env, N, NUM_STATES, NUM_ACTIONS, use_test_monte_carlo=True, do_memorize_cards=False):
    rewards = []
    Q_values = np.zeros(shape=(NUM_STATES, NUM_ACTIONS))
    returns = [[[] for i in range(NUM_ACTIONS)] for j in range(NUM_STATES)]

    for i in trange(N_iters):
    
        epsilon = EPSILON_INIT - i * (EPSILON_INIT - EPSILON_END) / N_iters
        # generate episode
        episode = []
        (player_score, dealer_card, usable_ace), _ = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(Q_values[player_score])
            else:
                action = random.randint(0, 1)
            (player_score_next, dealer_card, usable_ace), reward, done, _, _ = env.step(action)
            episode.append((player_score, action, reward, done))
            player_score = player_score_next

        G = 0
        for player_score, action, reward, done in episode[::-1]:
            G = reward + G
            if not returns[player_score][action]:
                returns[player_score][action] = (G, 1)
            else:
                value, count = returns[player_score][action]
                returns[player_score][action] = ((G + value * count) / (count + 1), count + 1)

            Q_values[player_score, action] = returns[player_score][action][0]


        if i % CONTROL_INTERVAL == 0:
            def test_monte_carlo(inner_env, Q_values):
                (player_score, dealer_card, usable_ace), _ = inner_env.reset()
                done = False
                while not done:
                    if random.uniform(0, 1) > epsilon:
                        action = np.argmax(Q_values[player_score])
                    else:
                        action = random.randint(0, 1)
                    (player_score_next, dealer_card, usable_ace), reward, done, _, _ = inner_env.step(action)
                    player_score = player_score_next
                return reward
            
            def memorize_cards(inner_env, Q_values):
                (player_score, dealer_card, _), card_count_ = inner_env.reset()
                idx = get_index_from_card_count(card_count_)
                done = False
                while not done:
                    if random.uniform(0, 1) > epsilon:
                        action = np.argmax(Q_values[idx][player_score])
                    else:
                        action = random.randint(0, 1)
                    (player_score_next, next_dealer_card, _), reward, done, _, next_card_count_ = env.step(action)
                    player_score = player_score_next
                    dealer_card = next_dealer_card
                    idx = get_index_from_card_count(next_card_count_)
                return reward
            
            if use_test_monte_carlo:
                average_reward = np.mean(Parallel(n_jobs=-1)(delayed(test_monte_carlo)(env, Q_values) for _ in range(N)))
            if do_memorize_cards:
                average_reward = np.mean(Parallel(n_jobs=-1)(delayed(memorize_cards)(env, Q_values) for _ in range(N)))
            rewards.append(average_reward)
            
    return rewards


def get_average_rewards_with_memorization(env, N, NUM_STATES, NUM_ACTIONS, MAX_INDEXES, use_test_monte_carlo=True, do_memorize_cards=False):
    rewards = []
    Q_values = np.zeros(shape=(MAX_INDEXES, NUM_STATES, NUM_ACTIONS))
    returns = [[[[]  for i in range(NUM_ACTIONS)] for j in range(NUM_STATES)] for k in range(MAX_INDEXES)]
    for i in trange(N_iters):
    
        epsilon = EPSILON_INIT - i * (EPSILON_INIT - EPSILON_END) / N_iters

        episode = []
        (player_score, dealer_card, _), card_count_ = custom_env.reset()
        idx = get_index_from_card_count(card_count_)
        done = False
        while not done:
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(Q_values[idx][player_score])
            else:
                action = random.randint(0, 1)
            (player_score_next, dealer_card, _), reward, done, _, next_card_count_ = custom_env.step(action)
            episode.append((player_score, action, reward, done, index_))
            player_score = player_score_next
            idx = get_index_from_card_count(next_card_count_)

        G = 0
        for player_score, action, reward, done, index_ in episode[::-1]:
            G = reward + G


            if not returns[idx][player_score][action]:
                returns[idx][player_score][action] = (G, 1)
            else:
                value, count = returns[idx][player_score][action]
                value = (value * count + G) / (count + 1)
                returns[idx][player_score][action] = (value, count + 1)

            Q_values[idx][player_score, action] =\
                returns[idx][player_score][action][0]


        # testing model  every 5000 iterations
        if i%CONTROL_INTERVAL==0:
            def memorize_cards(inner_env, Q_values):
                (player_score, dealer_card, _), card_count_ = inner_env.reset()
                idx = get_index_from_card_count(card_count_)
                done = False
                while not done:
                    if random.uniform(0, 1) > epsilon:
                        action = np.argmax(Q_values[idx][player_score])
                    else:
                        action = random.randint(0, 1)
                    (player_score_next, next_dealer_card, _), reward, done, _, next_card_count_ = env.step(action)
                    player_score = player_score_next
                    dealer_card = next_dealer_card
                    idx = get_index_from_card_count(next_card_count_)
                return reward
            
            average_reward = np.mean(Parallel(n_jobs=-1)(delayed(memorize_cards)(env, Q_values) for _ in range(N)))
            rewards.append(average_reward)
            
    return rewards