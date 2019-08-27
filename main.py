from datetime import datetime
from poker.game import Game
from tqdm import tqdm

if __name__ == '__main__':

    print(datetime.now().strftime('%H:%M:%S'))
    for _ in tqdm(range(10)):
        run = Game(n_players=4, simulation_iterations=5, parallelise=False)
        run.deal_hole()
        run.simulate()
        print(run.user.hand, run.user.win_probability)
        run.deal_community(n_cards=3)
        run.simulate()
        print(run.user.hand, run.user.win_probability)
        run.deal_community(n_cards=1)
        run.simulate()
        print(run.user.hand, run.user.win_probability)
        run.deal_community(n_cards=1)
        print('Final hand:\n', run.user.hand)
        winner = run.determine_winner()
        print(winner)
    print(datetime.now().strftime('%H:%M:%S'))