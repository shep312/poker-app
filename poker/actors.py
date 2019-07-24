import pandas as pd
import numpy as np
from poker.utils import poker_hands_rank, stage_names


class Player:
    
    def __init__(self):
        self.table_position = 0
        self.is_dealer = False
        self.is_small_blind = False
        self.is_big_blind = False
        self.folded = False
        self.hole = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.hand = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.hand_score = pd.DataFrame({
            'hand': list(poker_hands_rank.keys()),
            'hand_rank': list(poker_hands_rank.values()),
            'present': np.zeros([len(poker_hands_rank), ], dtype=bool),
            'probability_of_occurring': np.zeros([len(poker_hands_rank), ]),
            'high_card': np.zeros([len(poker_hands_rank), ], dtype=int)
        }).set_index('hand')
        self.best_hand = None
        self.best_hand_high_card = None
        
    def determine_hand(self, n_players):
        """
        Determine what hands are present in a player's hand
        and therefore how strong it is
    
        Args:
             hand (pandas.DataFrame): The hand as a dataframe
             n_players (int): Number of players in game    
        """
        assert isinstance(n_players, int), 'n_players must be a integer'
    
        # Determine number of cards left in the deck for probability calculations
        stage = stage_names.get(self.hand.shape[0], 'unknown')
        if stage != 'not_started':
            n_hole_cards = 2
            n_community_cards = max(self.hand.shape[0] - n_hole_cards, 0)
            cards_in_deck = 52 - n_community_cards - n_players * 2
        else:
            cards_in_deck = 52
    
        # Run checks to see what hands are currently present
        # High card
        if self.hand.shape[0]:
            self.hand_score.loc['High card', 'present'] = True
            self.hand_score.loc['High card', 'probability_of_occurring'] = 1
            self.hand_score.loc['High card', 'high_card'] = \
                self.hand['value'].max()
        else:
            self.hand_score.loc['High card', 'probability_of_occurring'] = 1
    
        # Pair
        value_counts = self.hand['value'].value_counts()
        if any(value_counts == 2):
            self.hand_score.loc['Pair', 'present'] = True
            self.hand_score.loc['Pair', 'probability_of_occurring'] = 1
            self.hand_score.loc['Pair', 'high_card'] = \
                value_counts[value_counts == 2].index.max()
        else:
            #TODO probability calc
            pass
    
        # Two pair
        if sum(value_counts == 2) == 2:
            self.hand_score.loc['Two pairs', 'present'] = True
            self.hand_score.loc['Two pairs', 'probability_of_occurring'] = 1
            self.hand_score.loc['Two pairs', 'high_card'] = \
                value_counts[value_counts == 2].index.max()
        else:
            #TODO probability calc
            pass
            
        # Three of a kind
        if any(value_counts == 3):
            self.hand_score.loc['Three of a kind', 'present'] = True
            self.hand_score.loc['Three of a kind', 'probability_of_occurring'] = 1
            self.hand_score.loc['Three of a kind', 'high_card'] = \
                value_counts[value_counts == 2].index.max()
        else:
            #TODO probability calc
            pass
        
        # Straight
        sorted_hand = self.hand.sort_values(by='value')
        sorted_hand['diff'] = sorted_hand['value'].diff()
        sorted_hand['not_linked'] = (sorted_hand['diff'] != 1).cumsum()
        sorted_hand['streak'] = sorted_hand.groupby('not_linked').cumcount()
        if sorted_hand['streak'] >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            self.hand_score.loc['Straight', 'high_card'] = \
                sorted_hand.loc[sorted_hand['streak'] == 4, 'value']
        else:
            #TODO probability calc
            pass

class Opponent(Player):
    pass

class User(Player):
    pass