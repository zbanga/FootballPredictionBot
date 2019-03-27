import pandas as pd
from Football Predictive Bot import season19

print(season19)
''' State Tracker class for keeping track of current states in the frontend '''
class StateTracker:

    def __init__(self, verbose=True): #Verbose means using or expressed in more words than needed 
        # set the initial states
        self.state = {
            'HomeTeam' : '', 
            'AwayTeam': '',
            'result': '',

        }

    ''' Updates the current state '''
    def updateState(self, request):
        self.state['HomeTeam'] = request.form['HomeTeam']
        self.state['AwayTeam'] = request.form['AwayTeam']
