from poker.game import Game

run = Game(n_players=4)
run.deal_hole()
run.deal_community(n_cards=3)
run.deal_community(n_cards=1)
run.deal_community(n_cards=1)