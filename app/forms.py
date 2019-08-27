from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField
from poker.utils import VALUES, SUITS
from wtforms.validators import DataRequired, NumberRange


value_choices = [(key, val) for key, val in VALUES.items()]
suit_choices = [(key, val) for key, val in SUITS.items()]


class HoleForm(FlaskForm):
    n_players = IntegerField(
        label='Number of players',
        validators=[DataRequired(), NumberRange(min=2, max=8)],
        default=4
    )

    card_1_values = SelectField('Card 1 value', choices=value_choices)
    card_1_suit = SelectField('Card 1 suit', choices=suit_choices)

    card_2_values = SelectField('Card 2 value', choices=value_choices)
    card_2_suit = SelectField('Card 2 suit', choices=suit_choices)

    submit = SubmitField('Calculate odds')