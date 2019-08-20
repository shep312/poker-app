from datetime import datetime
from poker.game import Game

print(datetime.now().strftime('%H:%M:%S'))
for _ in range(10):
    run = Game(n_players=4)
    run.deal_hole()
    run.simulate()
    run.deal_community(n_cards=3)
    run.deal_community(n_cards=1)
    run.deal_community(n_cards=1)
    run.determine_winner()
print(datetime.now().strftime('%H:%M:%S'))