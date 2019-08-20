import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from random import shuffle
from poker.actors import User, Opponent
from poker.utils import get_card_name, SUITS, VALUES


class Game:

    def __init__(self, n_players):
        assert n_players >= 3, 'Must be at least 3 players'
        self.n_players = n_players
        self.opponents = [Opponent() for i in range(n_players - 1)]
        self.user = User()
        self.players = self.opponents + [self.user]
        self.deck = []
        self.community_cards = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.pot = 0

        # Assign positions to players randomly
        positions = np.arange(n_players)
        shuffle(positions)
        for i, player in zip(positions, self.players):
            player.table_position = i

        # Sort by position
        self.players.sort(key=lambda x: x.table_position)

        # Assign initial roles
        self.players[0].is_dealer = True
        self.players[1].is_small_blind = True
        self.players[2].is_big_blind = True

        self.prepare_deck()

    def prepare_deck(self):
        value_names = list(VALUES.keys())
        suit_names = list(SUITS.keys())
        self.deck = [[suit, val] for val in value_names for suit in suit_names]
        shuffle(self.deck)

    def rotate_dealer(self):
        pass

    def deal_card(self, recipient=None):
        card = self.deck[-1]
        card_dict = {
            'suit': card[0],
            'value': card[1],
            'name': get_card_name(card)
        }
        if recipient:
            recipient.hole = \
                recipient.hole.append(card_dict, ignore_index=True)
            recipient.hole.reset_index(drop=True, inplace=True)
        else:
            self.community_cards = \
                self.community_cards.append(card_dict, ignore_index=True)
            self.community_cards.reset_index(drop=True, inplace=True)
        self.deck.remove(card)

    def deal_hole(self):
        for player in self.players:
            for _ in range(2):
                self.deal_card(player)
            player.hand = player.hole.copy()
        for player in self.players:
            player.determine_hand(n_players=self.n_players)
        self.simulate()

    def deal_community(self, n_cards=1):
        for _ in range(n_cards):
            self.deal_card()
        for player in self.players:
            if not player.folded:
                player.hand = pd.concat([player.hole, self.community_cards])
                player.hand.reset_index(drop=True, inplace=True)
        for player in self.players:
            player.determine_hand(n_players=self.n_players)
            
    def simulate(self, n_samples=100):
        card_frequencies = pd.DataFrame(
            index=self.user.hand_score.index,
            columns=['frequency'],
            data=np.zeros([self.user.hand_score.shape[0], ])
        )
        n_cards_left = 7 - self.user.hand.shape[0]
        user_wins = 0
        for i in tqdm(range(n_samples)):
            sim_game = deepcopy(self)
            shuffle(sim_game.deck)
            sim_game.deal_community(n_cards=n_cards_left)
            card_frequencies['frequency'] \
                += sim_game.user.hand_score['present'].astype(int)
            if self.determine_winner() == sim_game.user.table_position:
                user_wins += 1
            
        card_frequencies /= n_samples
        user_wins /= n_samples
        self.user.hand_score['probability_of_occurring'] = card_frequencies
        self.user.win_probability = user_wins

    def determine_winner(self):
        result = pd.DataFrame({
            'hand_score': [player.best_hand_numeric for player in self.players],
            'hand_name': [player.best_hand for player in self.players],
            'hand_high_card': [player.best_hand_high_card
                               for player in self.players],
            'position': [player.table_position for player in self.players]
        })
        result.sort_values(by='hand_score', ascending=False, inplace=True)
        winning_position = result.iloc[0, -1]
        return winning_position
