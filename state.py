import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from joblib import dump, load
import pickle
from sklearn.preprocessing import scale

model = pickle.load(open(r"C:\Users\Mlej\futbol-prediction-bot\LogisticRegression.pkl","rb"))

season19 = pd.read_csv(r"C:\Users\Mlej\futbol-prediction-bot\csv\statsPerEPLTeam.csv")



def newDF(stateTracker):
    home = stateTracker.HomeTeam
    away = stateTracker.AwayTeam  
    df = pd.DataFrame({'HomeTeam' : home,'AwayTeam': away}, index=[0])     
    return df.reset_index(drop=True)
 
def output(number, dataFrame):
    df = dataFrame
    if number == 1:
        x = df.iloc[0,0]
        return x
    elif number == -1:
        x = df.iloc[0,1]  
        return x
    else:
        x = 'Draw'
        return x


def preprocess(dataframe):
    data = pd.DataFrame(columns=['HomeTeam', 'AwayTeam'], index=[0])
    data = dataframe
    
    HomeTeam = data.iloc[0,0]
    AwayTeam = data.iloc[0,1]
       
    data['HAS'] = season19[season19.Team == HomeTeam]['HAS'].reset_index(drop=True)
    data['HDS'] = season19[season19.Team == HomeTeam]['HDS'].reset_index(drop=True)
    data['AAS'] = season19[season19.Team == AwayTeam]['AAS'].reset_index(drop=True)
    data['ADS'] = season19[season19.Team == AwayTeam]['ADS'].reset_index(drop=True)
    data['HTYCI'] = season19[season19.Team==HomeTeam]['HTYCI'].reset_index(drop=True)
    data['ATYCI'] = season19[season19.Team==AwayTeam]['ATYCI'].reset_index(drop=True)
    data['HTRCI'] = season19[season19.Team==HomeTeam]['HTRCI'].reset_index(drop=True)
    data['ATRCI'] = season19[season19.Team==AwayTeam]['ATRCI'].reset_index(drop=True)
    data['HTSOTI'] = season19[season19.Team==HomeTeam]['HTSOTI'].reset_index(drop=True)
    data['ATSOTI'] = season19[season19.Team==AwayTeam]['ATSOTI'].reset_index(drop=True)
    data['HTFI'] = season19[season19.Team==HomeTeam]['HTFI'].reset_index(drop=True)
    data['ATFI'] = season19[season19.Team==AwayTeam]['ATFI'].reset_index(drop=True)
    data['HTSI'] = season19[season19.Team==HomeTeam]['HTSI'].reset_index(drop=True)
    data['ATSI'] = season19[season19.Team==AwayTeam]['ATSI'].reset_index(drop=True)
    data['HTCKI'] = season19[season19.Team==HomeTeam]['HTCKI'].reset_index(drop=True)
    data['ATCKI'] = season19[season19.Team==AwayTeam]['ATCKI'].reset_index(drop=True)
    data['HTCR'] = season19[season19.Team==HomeTeam]['HTCR'].reset_index(drop=True)
    data['ATCR'] = season19[season19.Team==AwayTeam]['ATCR'].reset_index(drop=True)

    return data

def predict(data):
    predictData = data[['HAS', 'HDS', 'HTYCI', 'HTRCI', 'HTSOTI', 'HTFI', 'HTSI',
        'HTCKI', 'HTCR', 'AAS', 'ADS', 'ATYCI', 'ATRCI', 'ATSOTI', 'ATFI',
         'ATSI', 'ATCKI', 'ATCR']]            
    prediction = model.predict(predictData)
    return prediction[0]



''' State Tracker class for keeping track of current states in the frontend '''
class StateTracker:
    def __init__(self): #Verbose means using or expressed in more words than needed 
        # set the initial states
        self.HomeTeam = 'Burnley'
        self.AwayTeam = 'Bournemouth'
        
    
    def updateState(self, request):
        self.HomeTeam = request.form['HomeTeam']
        self.AwayTeam = request.form['AwayTeam']








