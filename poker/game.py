import numpy as np
import pandas as pd
from copy import deepcopy
from random import shuffle
from multiprocessing import Pool, cpu_count
from itertools import repeat
from poker.actors import User, Opponent
from poker.utils import get_card_name, SUITS, VALUES


def simulate(game):
    """
    Simulate completion of a game from its current state.
    
    PARAMETERS
    ----------
    game : poker.Game
        An instance of the Game class
        
    RETURNS
    -------
    user_wins : bool
        Whether or not the user won the game in this simulation
    user_draws : bool
        Whether or not the user drew the game in this simulation
    hand_occurrences : pandas.Series
        A series indexed by hand name that indicates which hands the user
        gets by the end of the simulation
    """
    
    # Copy the game object to modify in the simulation
    sim_game = deepcopy(game)
    
    # Initialise outputs
    user_wins, user_draws = False, False
    
    # Get the current state of the game
    n_cards_left = 7 - sim_game.user.hand.shape[0]
    users_cards = [[row['suit'], row['value']]
                   for _, row in sim_game.user.hand.iterrows()]
    
    # Reset the deck and opponents card to allow randomness for the monte
    # carlo sims
    sim_game.prepare_deck(excluded_cards=users_cards)
    sim_game.deal_hole(opponents_only=True)

    # Complete the rest of the game and save results
    sim_game.deal_community(n_cards=n_cards_left)
    hand_occurences \
        = sim_game.user.hand_score['present'].astype(int)
    winners = sim_game.determine_winner()
    
    # Check results
    if sim_game.user in winners:
        if len(winners) == 1:
            user_wins = True
        else:
            user_draws = True
    return user_wins, user_draws, hand_occurences


class Game:

    def __init__(self, n_players, simulation_iterations=10, parallelise=False):
        assert n_players >= 2, 'Must be at least 2 players'
        self.n_players = n_players
        self.n_iter = simulation_iterations
        self.opponents = [Opponent() for _ in range(n_players - 1)]
        self.user = User()
        self.players = self.opponents + [self.user]
        self.deck = []
        self.community_cards = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.pot = 0
        self.n_cores = cpu_count() if parallelise else None

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
        if self.n_cores:
            with Pool(processes=self.n_cores) as pool:
                results = \
                    pool.map(simulate, repeat(self, self.n_iter))
        else:
            results = map(simulate, repeat(self, self.n_iter))

        # Initialise variables to store results
        card_frequencies = pd.DataFrame(
            index=self.user.hand_score.index,
            columns=['frequency'],
            data=np.zeros([self.user.hand_score.shape[0], ])
        )
        user_wins, user_draws = 0, 0
        
        # Get results from the simulations
        for result in results:
            user_wins += result[0]
            user_draws += result[1]
            card_frequencies['frequency'] += result[2]

        # Return results to the user object
        self.user.hand_score['probability_of_occurring'] = \
            card_frequencies['frequency'] / self.n_iter
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
