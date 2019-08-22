import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from random import shuffle
from poker.actors import User, Opponent
from poker.utils import get_card_name, SUITS, VALUES


class Game:

    def __init__(self, n_players):
        assert n_players >= 2, 'Must be at least 2 players'
        self.n_players = n_players
        self.opponents = [Opponent() for _ in range(n_players - 1)]
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

    def prepare_deck(self, excluded_cards=None):
        value_names = list(VALUES.keys())
        suit_names = list(SUITS.keys())
        self.deck = [[suit, val] for val in value_names for suit in suit_names]
        if excluded_cards:
            for card in excluded_cards:
                self.deck.remove(card)
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

    def deal_hole(self, opponents_only=False):
        recipients = self.opponents if opponents_only else self.players
        for player in recipients:
            if opponents_only:
                player.hole = pd.DataFrame(columns=['suit', 'value', 'name'])
            for _ in range(2):
                self.deal_card(player)
            player.hand = player.hole.copy()
        for player in recipients:
            player.determine_hand()

    def deal_community(self, n_cards=1):
        for _ in range(n_cards):
            self.deal_card()
        for player in self.players:
            if not player.folded:
                player.hand = pd.concat([player.hole, self.community_cards])
                player.hand.reset_index(drop=True, inplace=True)
        for player in self.players:
            player.determine_hand()
            
    def simulate(self, n_samples=200):
        """
        Perform Monte Carlo simulation of remaining game from this point to
        determining:
            1) The probability of the user obtaining each hand
            2) The probability that the user will win the game

        PARAMETERS
        ----------
        n_samples : int
            Number of simulations to run
        """

        # Initialise somewhere to store the results of each simulation
        card_frequencies = pd.DataFrame(
            index=self.user.hand_score.index,
            columns=['frequency'],
            data=np.zeros([self.user.hand_score.shape[0], ])
        )
        user_wins = 0

        # Number of deals left in the game at this stage
        n_cards_left = 7 - self.user.hand.shape[0]

        # Loop through each simulation
        for _ in tqdm(range(n_samples)):

            # Need to randomise the contents on the deck and opponents hands
            sim_game = deepcopy(self)
            users_cards = [[row['suit'], row['value']]
                           for _, row in sim_game.user.hand.iterrows()]
            sim_game.prepare_deck(excluded_cards=users_cards)
            sim_game.deal_hole(opponents_only=True)

            # Complete the rest of the game and save results
            sim_game.deal_community(n_cards=n_cards_left)
            card_frequencies['frequency'] \
                += sim_game.user.hand_score['present'].astype(int)
            winning_player = sim_game.determine_winner()
            if winning_player is sim_game.user:
                user_wins += 1

        # Turn into probabilities and assign results
        card_frequencies /= n_samples
        user_wins /= n_samples
        self.user.hand_score['probability_of_occurring'] = card_frequencies
        self.user.win_probability = user_wins

    def determine_winner(self):

        # Compare each players best hand
        player_scores = pd.DataFrame(dict(
            player_position=np.zeros([len(self.players), ], dtype=int),
            hand_score=np.zeros([len(self.players), ], dtype=float)
        ))
        for i, player in enumerate(self.players):
            hands_made = player.hand_score[player.hand_score['present']]
            hands_made.sort_values(by=['hand_rank', 'high_card'],
                                   ascending=[False, False],
                                   inplace=True)

            player_scores.loc[i, 'player_position'] = \
                player.table_position
            player_scores.loc[i, 'hand_score'] = \
                hands_made.iloc[0, 0] + hands_made.iloc[0, -2] / 100

        player_scores = \
            player_scores.sort_values(by='hand_score', ascending=False)\
            .reset_index(drop=True)
        best_score = player_scores.loc[0, 'hand_score']
        score_shared = (player_scores['hand_score'] == best_score).sum() > 1

        # In case of draw: best hand
        if score_shared:
            tied_positions = \
                player_scores.loc[player_scores['hand_score'] == best_score,
                                  'player_position']
            for player in self.players:
                if player.table_position in tied_positions:
                    pass

        self.players.sort(key=lambda x: x.hand_score_numeric)
        return self.players[0]
