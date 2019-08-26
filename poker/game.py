import numpy as np
import pandas as pd
from copy import deepcopy
from random import shuffle
from multiprocessing import Pool, cpu_count
from poker.actors import User, Opponent
from poker.utils import get_card_name, SUITS, VALUES, WinnerNotFoundException

def simulate(game):
    sim_game = deepcopy(game)
    user_wins, user_draws = False, False
    n_cards_left = 7 - sim_game.user.hand.shape[0]

    users_cards = [[row['suit'], row['value']]
                   for _, row in sim_game.user.hand.iterrows()]
    sim_game.prepare_deck(excluded_cards=users_cards)
    sim_game.deal_hole(opponents_only=True)

    # Complete the rest of the game and save results
    sim_game.deal_community(n_cards=n_cards_left)
    # card_frequencies['frequency'] \
    #     += sim_game.user.hand_score['present'].astype(int)
    winners = sim_game.determine_winner()
    if sim_game.user in winners:
        if len(winners) == 1:
            user_wins = True
        else:
            user_draws = True
    return user_wins, user_draws

class Game:

    def __init__(self, n_players, simulation_iterations):
        assert n_players >= 2, 'Must be at least 2 players'
        self.n_players = n_players
        self.n_iter = simulation_iterations
        self.opponents = [Opponent() for _ in range(n_players - 1)]
        self.user = User()
        self.players = self.opponents + [self.user]
        self.deck = []
        self.community_cards = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.pot = 0
        self.n_cores = cpu_count()

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
        if self.n_players > 2:
            self.players[2].is_big_blind = True
        else:
            self.players[0].is_big_blind = True

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
            
    def simulate(self):
        """
        Perform Monte Carlo simulation of remaining game from this point to
        determining:
            1) The probability of the user obtaining each hand
            2) The probability that the user will win the game
        """

        # Loop through each simulation
        with Pool(processes=self.n_cores) as pool:
            results = pool.map(simulate, [self for _ in range(self.n_iter)])

        print('Got results back')
        user_wins, user_draws = 0, 0
        for result in results:
            user_wins += result[0]
            user_draws += result[1]

        # self.user.hand_score['probability_of_occurring'] = card_frequencies
        self.user.win_probability = user_wins / self.n_iter
        self.user.draw_probability = user_draws / self.n_iter

    def determine_winner(self):
        for player in self.players:
            player.get_best_five_card_hand()
        
        maximum_number_of_hands = 5        
        for i in range(maximum_number_of_hands):
            
            # Check best hand
            scores = []
            for player in self.players:
                if len(player.hand_scores_numeric) > i:
                    scores.append(player.hand_scores_numeric[i])
                else:
                    scores.append(0)
            max_score = max(scores)
            n_max_score = sum(max_score == np.array(scores))
            
            # Fold those out at this stage
            for j, player in enumerate(self.players):
                if scores[j] < max_score:
                    player.fold()
                    
            # If there is a standalone winner then exit now
            if n_max_score == 1:
                break
        
        return [player for player in self.players if not player.folded]
