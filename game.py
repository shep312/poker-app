import numpy as np
import pandas as pd
from random import shuffle
from actors import User, Opponent, Player


class Game:

    def __init__(self, n_players):
        assert n_players >= 3, 'Must be at least 3 players'
        self.n_players = n_players
        self.opponents = [Opponent() for i in range(n_players - 1)]
        self.user = User()
        self.players = self.opponents + [self.user]
        self.deck = []
        self.community_cards = pd.DataFrame(columns=['suit', 'value'])
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

        self.shuffle()


    def shuffle(self):
        suits = (0, 1, 2, 3)
        values = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
        self.deck = [[val, suit] for val in values for suit in suits]
        shuffle(self.deck)

    def rotate_dealer(self):
        pass

    def deal_card(self, player):
        card = self.deck[-1]
        if isinstance(player, Player):
            player.hand.loc[:-1, :] = card
        elif isinstance(player, pd.DataFrame):
            player.loc[:-1, :] = card
        else:
            raise TypeError('Entity being dealt to must be a DataFrame')
        self.deck.remove(card)

    def deal_hole(self):
        for i in range(2):
            for player in self.players:
                self.deal_card(player)

    def deal_community(self, n_cards=1):
        for i in range(n_cards):
            self.deal_card(self.community_cards)
        for player in self.players:
            player.hand += self.community_cards

