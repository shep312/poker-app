import pandas as pd

class Player:
    def __init__(self):
        self.hand = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.hand_score = None
        self.table_position = 0
        self.is_dealer = False
        self.is_small_blind = False
        self.is_big_blind = False
        self.folded = False

class Opponent(Player):
    pass

class User(Player):
    pass