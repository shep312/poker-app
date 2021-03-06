from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField
from poker.utils import VALUES, SUITS
from wtforms.validators import DataRequired, NumberRange


value_choices = [(key, val) for key, val in VALUES.items()]
suit_choices = [(key, val) for key, val in SUITS.items()]

dropdown_style = {'class': 'button'}
input_style = {'class': 'button'}


class HoleForm(FlaskForm):
    n_players = IntegerField(
        label='Number of players',
        validators=[DataRequired(), NumberRange(min=2, max=8)],
        default=6,
        render_kw=input_style
    )

    card_1_values = SelectField('Card 1 value',
                                choices=value_choices,
                                render_kw=dropdown_style)
    card_1_suit = SelectField('Card 1 suit',
                              choices=suit_choices,
                              render_kw=dropdown_style)

    card_2_values = SelectField('Card 2 value',
                                choices=value_choices,
                                render_kw=dropdown_style)
    card_2_suit = SelectField('Card 2 suit',
                              choices=suit_choices,
                              render_kw=dropdown_style)

    submit = SubmitField('Calculate odds', render_kw=input_style)


class FlopForm(FlaskForm):
    card_1_values = SelectField('Card 1 value',
                                choices=value_choices,
                                render_kw=dropdown_style)
    card_1_suit = SelectField('Card 1 suit',
                              choices=suit_choices,
                              render_kw=dropdown_style)

    card_2_values = SelectField('Card 2 value',
                                choices=value_choices,
                                render_kw=dropdown_style)
    card_2_suit = SelectField('Card 2 suit',
                              choices=suit_choices,
                              render_kw=dropdown_style)

    card_3_values = SelectField('Card 3 value',
                                choices=value_choices,
                                render_kw=dropdown_style)
    card_3_suit = SelectField('Card 3 suit',
                              choices=suit_choices,
                              render_kw=dropdown_style)

    submit = SubmitField('Calculate odds')


class TurnOrRiverForm(FlaskForm):
    card_1_values = SelectField('Card 1 value',
                                choices=value_choices,
                                render_kw=dropdown_style)
    card_1_suit = SelectField('Card 1 suit',
                              choices=suit_choices,
                              render_kw=dropdown_style)

    submit = SubmitField('Calculate odds')