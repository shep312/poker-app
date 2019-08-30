from app import app
from poker.game import Game
from flask import render_template, request, redirect
from app.forms import HoleForm, FlopForm, TurnOrRiverForm


@app.route('/', methods=['GET', 'POST'])
def index():
    form = HoleForm()

    if request.method == 'POST':
        app.config['game'] = Game(n_players=form.n_players.data,
                                  simulation_iterations=150,
                                  parallelise=True)
        card_1 = [int(form.card_1_suit.data), int(form.card_1_values.data)]
        card_2 = [int(form.card_2_suit.data), int(form.card_2_values.data)]
        users_cards = [card_1, card_2]
        for card in users_cards:
            app.config['game'].deal_card(recipient=app.config['game'].user,
                                         card=card)
        app.config['game'].simulate()
        return redirect('hole')

    return render_template('index.html', form=form)


@app.route('/hole', methods=['GET', 'POST'])
def hole():
    form = FlopForm()
    if request.method == 'POST':
        flop_cards = [
            [int(form.card_1_suit.data), int(form.card_1_values.data)],
            [int(form.card_2_suit.data), int(form.card_2_values.data)],
            [int(form.card_3_suit.data), int(form.card_3_values.data)]
        ]
        app.config['game'].deal_community(cards=flop_cards)
        app.config['game'].simulate()
        return redirect('flop')

    probabilities = \
        app.config['game'].user.hand_score['probability_of_occurring']\
            .reset_index().to_html(index=False)
    return render_template(
        'hole.html',
        hand=app.config['game'].user.hand.to_html(),
        hand_probs=probabilities,
        win_prob=app.config['game'].user.win_probability * 100,
        form=form
    )


@app.route('/flop', methods=['GET', 'POST'])
def flop():
    form = TurnOrRiverForm()
    if request.method == 'POST':
        turn_card = \
            [[int(form.card_1_suit.data), int(form.card_1_values.data)]]
        app.config['game'].deal_community(cards=turn_card)
        app.config['game'].simulate()
        return redirect('turn')
        
    probabilities = \
        app.config['game'].user.hand_score['probability_of_occurring']\
            .reset_index().to_html(index=False)
    return render_template(
        'flop.html',
        hand=app.config['game'].user.hand.to_html(),
        hand_probs=probabilities,
        win_prob=app.config['game'].user.win_probability * 100,
        form=form
    )
    
    
@app.route('/turn', methods=['GET', 'POST'])
def turn():
    form = TurnOrRiverForm()
    if request.method == 'POST':
        turn_card = \
            [[int(form.card_1_suit.data), int(form.card_1_values.data)]]
        app.config['game'].deal_community(cards=turn_card)
        app.config['game'].simulate()
        return redirect('river')
        
    probabilities = \
        app.config['game'].user.hand_score['probability_of_occurring']\
            .reset_index().to_html(index=False)
    return render_template(
        'flop.html',
        hand=app.config['game'].user.hand.to_html(),
        hand_probs=probabilities,
        win_prob=app.config['game'].user.win_probability * 100,
        form=form
    )
    
    
@app.route('/river', methods=['GET', 'POST'])
def river():
    form = TurnOrRiverForm()
        
    probabilities = \
        app.config['game'].user.hand_score['probability_of_occurring']\
            .reset_index().to_html(index=False)
    return render_template(
        'river.html',
        hand=app.config['game'].user.hand.to_html(),
        hand_probs=probabilities,
        win_prob=app.config['game'].user.win_probability * 100
    )