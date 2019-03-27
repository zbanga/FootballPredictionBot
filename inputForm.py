from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired


class inputTeamsForm(FlaskForm):
    HomeTeam = StringField('Home Team', validators = [DataRequired()])
    AwayTeam = StringField('Away Team', validators = [DataRequired()])
    Predict = SubmitField('Predict')
