from app import app
from poker.game import Game
from poker.utils import get_card_name
from flask import render_template, request
from app.forms import HoleForm


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = HoleForm()
    if request.method == 'POST':
        game = Game(n_players=form.n_players.data,
                    simulation_iterations=50,
                    parallelise=True)

        suit_1, val_1 = int(form.card_1_suit.data), int(form.card_1_values.data)
        card_1_dict = {
            'suit': suit_1,
            'value': val_1,
            'name': get_card_name((suit_1, val_1))
        }

        suit_2, val_2 = int(form.card_2_suit.data), int(form.card_2_values.data)
        card_2_dict = {
            'suit': suit_2,
            'value': val_2,
            'name': get_card_name((suit_2 ,val_2))
        }

        for card_dict in [card_1_dict, card_2_dict]:
            game.user.hole = \
                game.user.hole.append(card_dict, ignore_index=True)
        game.user.hole.reset_index(drop=True, inplace=True)
        game.user.hand = game.user.hole.copy()

        users_cards = [[row['suit'], row['value']]
                       for _, row in game.user.hand.iterrows()]
        game.prepare_deck(excluded_cards=users_cards)
        game.deal_hole(opponents_only=True)
        game.simulate()

        return render_template('hole.html',
                               hand=game.user.hand.to_html(),
                               hand_probs=game.user.hand_score.to_html(),
                               win_prob=game.user.win_probability * 100)
    return render_template('index.html', form=form)
