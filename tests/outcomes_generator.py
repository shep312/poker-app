from pathlib import Path
from numpy import arange
from pandas import DataFrame
from tqdm import tqdm
from poker.game import Game
from poker.utils import HANDS_RANK


N_RECORDS = 5000
OUT_DIR = Path('.') / 'tests' / 'test_data'
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_PATH = OUT_DIR / 'generated_outcomes.csv'

output_columns = ['card_1_value', 'card_1_suit', 'card_2_value',
                  'card_2_suit']
output_columns += [hand.lower().replace(' ', '_') + '_probability_at_hole'
                   for hand in HANDS_RANK.keys()]
output_columns += [hand.lower().replace(' ', '_') + '_occurred_in_game'
                   for hand in HANDS_RANK.keys()]
outcomes = DataFrame(columns=output_columns, index=arange(N_RECORDS))

for i, row in tqdm(outcomes.iterrows()):
    run = Game(n_players=4)
    run.deal_hole()

    # Record users hand at hole
    outcomes.loc[i, 'card_1_value'] = run.user.hand.loc[0, 'value']
    outcomes.loc[i, 'card_1_suit'] = run.user.hand.loc[0, 'suit']
    outcomes.loc[i, 'card_2_value'] = run.user.hand.loc[1, 'value']
    outcomes.loc[i, 'card_2_suit'] = run.user.hand.loc[1, 'suit']

    # Record probabilities
    probs = run.user.hand_score
    for hand in HANDS_RANK.keys():
        outcomes.loc[i, hand.lower().replace(' ', '_') + '_probability_at_hole'] = \
            probs.loc[hand, 'probability_of_occurring']

    # Complete game
    run.deal_community(n_cards=3)
    run.deal_community(n_cards=1)
    run.deal_community(n_cards=1)

    # Record outcomes
    probs = run.user.hand_score
    for hand in HANDS_RANK.keys():
        outcomes.loc[i, hand.lower().replace(' ', '_') + '_occurred_in_game'] = \
            probs.loc[hand, 'present']

outcomes.to_csv(OUT_PATH)