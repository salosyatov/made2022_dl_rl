# За основу взята реализация https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

import numpy as np
from gym.envs.toy_text import BlackjackEnv


def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(card_count):
    counts = [count_ for count_ in card_count.values()]
    keys = [key_ for key_ in card_count.keys()]
    probs = [count_ / sum(counts) for count_ in counts]
    return np.random.choice(keys, p=probs)


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

class BlackJackWithDeck(BlackjackEnv):
    def __init__(self):
        super(BlackJackWithDeck, self).__init__()
        self.card_count = {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 
                           7: 4, 8: 4, 9: 4, 10: 16, 1: 4}
        self.limit = 15
        
    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            card_cur = draw_card(self.card_count)
            self.card_count[card_cur] -= 1
            self.player.append(card_cur)
            
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            terminated = True
            while sum_hand(self.dealer) < 17:
                card_cur = draw_card(self.card_count)
                self.card_count[card_cur] -= 1
                self.dealer.append(card_cur)
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, self.card_count
    
    def draw_hand(self):
        card_first = draw_card(self.card_count)
        self.card_count[card_first] -= 1
        card_second = draw_card(self.card_count)
        self.card_count[card_second] -= 1
        return [card_first, card_second]
    
    def reset(self, seed = None, options = None):
        
        # reset deck
        if sum(self.card_count.values()) < self.limit:
            self.card_count = {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 
                           7: 4, 8: 4, 9: 4, 10: 16, 1: 4}
        
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
 

        _, dealer_card_value, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self.card_count
    
    
# Как и в системе счёта десяток, каждой карте, выходящей из колоды, присваивается своё числовое значение:

# Карты 	Числовые значения
# 2, 3, 4, 5, 6 	+1
# 7, 8, 9       	0
# 10, В, Д, К, Т	−1

ETALON = {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 16, 1: 4}
PRICES = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 1: -1}
def get_index_from_card_count(card_count):
    return 37 + sum([(ETALON[card] - card_count[card]) * PRICES[card] for card in card_count])