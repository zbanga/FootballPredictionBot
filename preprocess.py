import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pickle
from state import StateTracker

season19 = pd.read_csv(r"C:\Users\Mlej\futbol-prediction-bot\csv\statsPerEPLTeam.csv")

model = pickle.load(open("LogisticRegression.pkl","rb"))

new = StateTracker()
df = newDF(new)
preprocess(df)

def preprocess(DataFrame):
    #Still contains the teams 
    data = DataFrame
    # data = pd.DataFrame({'HomeTeam': 'Arsenal', 'AwayTeam': 'Chelsea'}, index=[0]  
    HAS = [1]
    HDS = [1]
    AAS = [1]
    ADS = [1]
    HTYCI = [1]
    ATYCI = [1]
    HTRCI = [1]
    ATRCI = [1]
    HTSOTI = [1]
    ATSOTI = [1]
    HTFI = [1]
    ATFI = [1]
    HTSI = [1]
    ATSI = [1]
    HTCKI = [1]
    ATCKI = [1]
    HTCR = [1]
    ATCR = [1]

    for index,row in data.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs. 
        HAS.append(season19[season19.Team == row['HomeTeam']]['HAS'].values[0])     #error here,Like the error says, row is a tuple, so you can't do row["pool_number"]. You need to use the index: row[0].
        HDS.append(season19[season19.Team == row['HomeTeam']]['HDS'].values[0])
        AAS.append(season19[season19.Team == row['AwayTeam']]['AAS'].values[0])
        ADS.append(season19[season19.Team == row['AwayTeam']]['ADS'].values[0])
        HTYCI.append(season19[season19.Team==row['HomeTeam']]['HTYCI'].values[0])
        ATYCI.append(season19[season19.Team==row['AwayTeam']]['ATYCI'].values[0])
        HTRCI.append(season19[season19.Team==row['HomeTeam']]['HTRCI'].values[0])
        ATRCI.append(season19[season19.Team==row['AwayTeam']]['ATRCI'].values[0])
        HTSOTI.append(season19[season19.Team==row['HomeTeam']]['HTSOTI'].values[0])
        ATSOTI.append(season19[season19.Team==row['AwayTeam']]['ATSOTI'].values[0])
        HTFI.append(season19[season19.Team==row['HomeTeam']]['HTFI'].values[0])
        ATFI.append(season19[season19.Team==row['AwayTeam']]['ATFI'].values[0])
        HTSI.append(season19[season19.Team==row['HomeTeam']]['HTSI'].values[0])
        ATSI.append(season19[season19.Team==row['AwayTeam']]['ATSI'].values[0])
        HTCKI.append(season19[season19.Team==row['HomeTeam']]['HTCKI'].values[0])
        ATCKI.append(season19[season19.Team==row['AwayTeam']]['ATCKI'].values[0])
        HTCR.append(season19[season19.Team==row['HomeTeam']]['HTCR'].values[0])
        ATCR.append(season19[season19.Team==row['AwayTeam']]['ATCR'].values[0])

    data['HAS']=HAS[0]
    data['HDS']=HDS[0]
    data['HTYCI']=HTYCI[0]
    data['HTRCI']=HTRCI[0]
    data['HTSOTI']=HTSOTI[0]
    data['HTFI']=HTFI[0]
    data['HTSI']=HTSI[0]
    data['HTCKI']=HTCKI[0]
    data['HTCR']=HTCR[0]
    data['AAS']=AAS[0]
    data['ADS']=ADS[0]
    data['ATYCI']=ATYCI[0]
    data['ATRCI']=ATRCI[0]
    data['ATSOTI']=ATSOTI[0]
    data['ATFI']=ATFI[0]
    data['ATSI']=ATSI[0]
    data['ATCKI']=ATCKI[0]
    data['ATCR']=ATCR[0]

    return data


def predict(data):
    predictData = data[['HAS', 'HDS', 'HTYCI', 'HTRCI', 'HTSOTI', 'HTFI', 'HTSI',
        'HTCKI', 'HTCR', 'AAS', 'ADS', 'ATYCI', 'ATRCI', 'ATSOTI', 'ATFI',
         'ATSI', 'ATCKI', 'ATCR']]
    #scale data
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(predictData)
    doubleArray = [scaledData]
    prediction = model.predict(doubleArray)
    return prediction[0]
