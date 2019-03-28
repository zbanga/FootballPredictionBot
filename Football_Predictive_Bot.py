#!/usr/bin/env python
# coding: utf-8

# **BPL SOCCER PREDICTIVE BOT**

# In[1]:


#importing all the libraries
import pandas as pd 
import xgboost as xgb
import numpy as np
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import svm
from sklearn.metrics import classification_report
from scipy import io
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import voting_classifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# **Introduction to XGBoost**
# 
# XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples. XGBoost is recognized for its speed and performance
# 
# Documentation:https://xgboost.readthedocs.io/en/latest/ 
# 
# https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/
# 
# **Note that my dataset "Premier League Stats" is updated weekly at :**
# http://www.football-data.co.uk/englandm.php

# In[2]:


df = pd.read_csv('Premier League Stats.csv')

df.info()
df.head()


# **Data Cleaning**
# 

# In[3]:


#Getting rid of all betting data from websites sur as Bet365, Betfair, etc.
desired_columns=['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']
BPLData = df[desired_columns]
print(BPLData.head(10))
#print(BPLData.tail())
print(BPLData.shape)
#df.drop(['Div','Date','Referee','B365H','B365D', 'B365A', 'BWH', 'BWD','BWA','IWH','IWD','IWA','PSH', 'PSD', 'PSA','WHH','WHD','WHA','VCH','VCD','VCA','Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx>2.5','BbAv>2.5','BbMx<2.5','BbAv<2.5','BbAH','bAHh','BbMxAHH','BbAvAHH','BbMxAHA','BbAvAHA','PSCH','PSCD','PSCA'], axis=1, inplace=True)

#Removed Division (useless), Data, Referee, and all betting data.


# **Data Exploration** 

# In[4]:


#Inspecting our dataset to find interesting insights 

n_features = BPLData.shape[1] #gives the number of features .shape[1] gives the number of columns (number of features)
n_matches = BPLData.shape[0] #gives the number of matches .shape[0] gives the number of sample in the dataset
print(str(n_features) + ' features')
print(str(n_matches) + ' matches ')

total_home_goal_per_team = BPLData.groupby('HomeTeam').FTHG.sum().reset_index()
#print(total_home_goal_per_team)#we can clearly see that Home teams score more goals than away teams on average

total_away_goal_per_team = BPLData.groupby('AwayTeam').FTAG.sum().reset_index()
#print(total_away_goal_per_team)

total_goal_conceded_per_home_team = BPLData.groupby('HomeTeam').FTAG.sum().reset_index()
#print(total_goal_conceded_per_home_team)

total_goal_conceded_per_away_team = BPLData.groupby('AwayTeam').FTHG.sum().reset_index()
#print(total_goal_conceded_per_away_team)

#Let's find the average home and away goals scored by each team
#First we have to find how many home and away game each team played ***Considering 
numberOfHomeGamesPerTeam = BPLData.groupby('HomeTeam').AwayTeam.count().reset_index()
#print(numberOfHomeGamesPerTeam)
#numberOfHomeGamesPerTeam.info()
numberOfAwayGamesPerTeam = BPLData.groupby('AwayTeam').HomeTeam.count().reset_index()
print(numberOfAwayGamesPerTeam)


#geting a list of the teams
arrayOfTeamsInOrder = BPLData.HomeTeam.sort_values().unique()
listOfTeamsInOrder = list(arrayOfTeamsInOrder)
#print(listOfTeamsInOrder)

#We divide by the number of home/away goals by the number of home/away games to find the average of 
total_home_goal_per_team['AverageHomeGoalsPerGame'] = (total_home_goal_per_team['FTHG']/numberOfHomeGamesPerTeam['AwayTeam'])
#print(total_home_goal_per_team)

plt.figure(figsize=(16, 5))
plt.bar(range(len(total_home_goal_per_team['AverageHomeGoalsPerGame'])),total_home_goal_per_team['AverageHomeGoalsPerGame'], Color='#0000cc')
plt.xlabel('Barclays Premier League Team')
plt.ylabel('Average Goals Per Game')
plt.title('Graph of Average Home Goals Per Game')
ax = plt.subplot(1,1,1)
ax.set_xticks(range(len(total_home_goal_per_team['AverageHomeGoalsPerGame'])))
ax.set_xticklabels(listOfTeamsInOrder, rotation = 50)
#plt.show()
#plt.savefig('AverageGoals.png')
#From the graph we can clearly see the higher performance of the top six teams (Manchester City, Manchester United, Arsenal, Chelsea, Tottenham, Liverpool)

total_away_goal_per_team['AverageAwayGoalsPerGame'] = (total_away_goal_per_team['FTAG']/numberOfAwayGamesPerTeam['HomeTeam'])
#print(total_away_goal_per_team)
plt.close('all')
plt.figure(figsize=(16, 5))
plt.bar(range(len(total_away_goal_per_team['AverageAwayGoalsPerGame'])),total_away_goal_per_team['AverageAwayGoalsPerGame'],Color='#9933ff')
plt.xlabel('Barclays Premier League Team')
plt.ylabel('Average Goals Per Game')
plt.title('Graph of Average Away Goals Per Game')
ax = plt.subplot(1,1,1)
ax.set_xticks(range(len(total_home_goal_per_team['AverageHomeGoalsPerGame'])))
ax.set_xticklabels(listOfTeamsInOrder, rotation = 50)
#plt.show()


#percentage of Home wins this season to see if home team advantage is true 
homeTeamStats = BPLData.groupby('FTR').HomeTeam.count().reset_index()
#print(homeTeamStats)  #We can clearly see that there's a higher amount of H (represents HomeTeam wins) than A (represents Away team), therefore it confirms our hypothesis that Home Team has a higher winning chance
#print(len(BPLData[BPLData.FTR == 'H']))

#Calculating Home Win Percentage
homeWinRate = (len(BPLData[BPLData.FTR == 'H'])/n_matches ) * 100
#print('Home Team Win Rate: ' + str(homeWinRate)) #Home team has a 47.4% winning rate compared to 33.2% from the Away team

awayWinRate = (len(BPLData[BPLData.FTR == 'A'])/n_matches ) * 100
#print('Away Team Win Rate: ' + str(awayWinRate))

#average goals by home team (without considering any particular team)
totalHomeGoals = BPLData.FTHG.sum()
averageHomeTeamGoal = float(totalHomeGoals) / n_matches 
#print("Average number of home goals :", averageHomeTeamGoal) #We can clearly see from this print statement that on average the home team scores more than the away team.

#average goals by away team (without considering any particular team)
totalAwayGoals = BPLData.FTAG.sum()
averageAwayTeamGoal = float(totalAwayGoals) / n_matches 
#print("Average number of away goals :", averageAwayTeamGoal)

#Inspecting every game of a single team, displaying every Man City games...
df = BPLData[(BPLData['HomeTeam'] == 'Man City') | (BPLData['AwayTeam'] == 'Man City')]
ManCity = df.iloc[:]
#print(ManCity.head())

averageHomeGoalsConceded = averageAwayTeamGoal
averageAwayGoalsConceded = averageHomeTeamGoal
#print("Average number of goals conceded at home",avg_home_conceded_16)
#print("Average number of goals conceded away",avg_away_conceded_16)




# **Feature Engineering**

# In[5]:


print(BPLData.head())
#BPLData.groupby('FTR').HomeTeam.count().reset_index() example 

#print(numberOfHomeGamesPerTeam)
#print(numberOfAwayGamesPerTeam)

#Finding an index for yellow cards for teams at home or away
totalYellowCardPerHomeTeam = BPLData.groupby('HomeTeam').HY.sum().reset_index()
#print(totalYellowCardPerHomeTeam)
totalYellowCardPerAwayTeam = BPLData.groupby('AwayTeam').AY.sum().reset_index()
#print(totalYellowCardPerAwayTeam)

totalYellowCardHome = BPLData['HY'].sum()
#print(totalYellowCardHome)
totalYellowCardAway = BPLData['AY'].sum()
#print(totalYellowCardAway)

averageYellowCardHome = totalYellowCardHome/n_matches
#print(averageYellowCardHome)
averageYellowCardAway = totalYellowCardAway/n_matches
#print(averageYellowCardAway)

yellowCardIndexPerHomeTeam = (totalYellowCardPerHomeTeam.HY/numberOfHomeGamesPerTeam.AwayTeam) / averageYellowCardHome
#print(yellowCardIndexPerHomeTeam)
yellowCardIndexPerAwayTeam = (totalYellowCardPerAwayTeam.AY/numberOfAwayGamesPerTeam.HomeTeam) / averageYellowCardAway
#print(yellowCardIndexPerAwayTeam)

#Finding an index for red cards for teams at home or away

totalRedCardPerHomeTeam = BPLData.groupby('HomeTeam').HR.sum().reset_index()
#print(totalRedCardPerAwayTeam)
totalRedCardPerAwayTeam = BPLData.groupby('AwayTeam').AR.sum().reset_index()
#print(totalRedCardPerAwayTeam)
totalRedCardHome = BPLData.HR.sum()
# print(totalRedCardHome)
totalRedCardAway = BPLData.AR.sum()
# print(totalRedCardAway)
averageRedCardHome = totalRedCardHome / n_matches
#print(averageRedCardHome)
averageRedCardAway = totalRedCardAway / n_matches 
#print(averageRedCardAway)

redCardIndexPerHomeTeam = (totalRedCardPerHomeTeam.HR/numberOfHomeGamesPerTeam.AwayTeam) / averageRedCardHome
#print(redCardIndexPerHomeTeam)
redCardIndexPerAwayTeam = (totalRedCardPerAwayTeam.AR/numberOfAwayGamesPerTeam.HomeTeam) / averageRedCardAway
#print(redCardIndexPerAwayTeam)

#Finding an index for the # of shots on target for teams at home or away

totalShotOnTargetPerHomeTeam = BPLData.groupby('HomeTeam').HST.sum().reset_index()
#print(totalShotOnTargetPerHomeTeam)
totalShotOnTargetPerAwayTeam = BPLData.groupby('AwayTeam').AST.sum().reset_index()
#print(totalShotOnTargetPerAwayTeam)
totalShotOnTargetHome = BPLData.HST.sum()
#print(totalShotOnTargetHome)
totalShotOnTargetAway = BPLData.AST.sum()
#print(totalShotOnTargetAway)

shotOnTargetIndexPerHomeTeam = (totalShotOnTargetPerHomeTeam.HST/numberOfHomeGamesPerTeam.AwayTeam) / (totalShotOnTargetHome/n_matches)
#print(shotOnTargetIndexPerHomeTeam)
shotOnTargetIndexPerAwayTeam = (totalShotOnTargetPerAwayTeam.AST/numberOfAwayGamesPerTeam.HomeTeam) / (totalShotOnTargetAway/n_matches)
#print(shotOnTargetIndexPerAwayTeam)

#Finding an index for the number of fouls 

totalFoulsPerHomeTeam = BPLData.groupby('HomeTeam').HF.sum().reset_index() 
#print(totalFoulsPerHomeTeam)
totalFoulsPerAwayTeam = BPLData.groupby('AwayTeam').AF.sum().reset_index() 
#print(totalFoulsPerAwayTeam)
totalHomeFouls = BPLData.HF.sum()
#print(totalHomeFouls)
totalAwayFouls = BPLData.AF.sum()
#print(totalAwayFouls)
foulIndexPerHomeTeam = (totalFoulsPerHomeTeam.HF/numberOfHomeGamesPerTeam.AwayTeam) / (totalHomeFouls/n_matches)
# print(foulIndexPerHomeTeam)
foulIndexPerAwayTeam = (totalFoulsPerAwayTeam.AF/numberOfAwayGamesPerTeam.HomeTeam) / (totalAwayFouls/n_matches)
# print(foulIndexPerAwayTeam)

#Finding an index for the number of shots 
totalShotsPerHomeTeam = BPLData.groupby('HomeTeam').HS.sum().reset_index()
#print(totalShotsPerHomeTeam)
totalShotsPerAwayTeam = BPLData.groupby('AwayTeam').AS.sum().reset_index()
#print(totalShotsPerAwayTeam)
totalHomeShots = BPLData.HS.sum()
#print(totalHomeShots)
totalAwayShots = BPLData.AS.sum()
#print(totalAwayShots)
shotIndexPerHomeTeam = (totalShotsPerHomeTeam.HS/numberOfHomeGamesPerTeam.AwayTeam) / (totalHomeShots/n_matches)
# print(shotIndexPerHomeTeam)
shotIndexPerAwayTeam = (totalShotsPerAwayTeam.AS/numberOfAwayGamesPerTeam.HomeTeam) / (totalAwayShots/n_matches)
# print(shotIndexPerAwayTeam)

#Finding an index for the number of cornerkicks
totalNumberOfCornerKicksPerHomeTeam = BPLData.groupby('HomeTeam').HC.sum().reset_index()
#print(totalNumberOfCornerKicksPerHomeTeam)
totalNumberOfCornerKicksPerAwayTeam = BPLData.groupby('AwayTeam').AC.sum().reset_index()
#print(totalNumberOfCornerKicksPerAwayTeam)
totalCornerKicksHome = BPLData.HC.sum()
#print(totalCornerKicksHome)
totalCornerKicksAway = BPLData.AC.sum()
#print(totalCornerKicksAway)
cornerKicksIndexPerHomeTeam = (totalNumberOfCornerKicksPerHomeTeam.HC/numberOfHomeGamesPerTeam.AwayTeam) / (totalCornerKicksHome/n_matches)
#print(cornerKicksIndexPerHomeTeam)
cornerKicksIndexPerAwayTeam = (totalNumberOfCornerKicksPerAwayTeam.AC/numberOfAwayGamesPerTeam.HomeTeam) / (totalCornerKicksAway/n_matches)
#print(cornerKicksIndexPerAwayTeam)

#Finding an index of goal efficiency or we can say Conversion Rate (Number of Goals / Number of Shots) VERY IMPORTANT tells us how a team is efficient in front of the goal
totalNumberOfGoalsPerHomeTeam = BPLData.groupby('HomeTeam').FTHG.sum().reset_index()
#print(totalNumberOfGoalsPerHomeTeam)
totalNumberOfGoalsPerAwayTeam = BPLData.groupby('AwayTeam').FTAG.sum().reset_index()
#print(totalNumberOfGoalsPerAwayTeam)
totalNumberOfShotsPerHomeTeam = BPLData.groupby('HomeTeam').HS.sum().reset_index()
#print(totalNumberOfShotsPerHomeTeam)
totalNumberOfShotsPerAwayTeam = BPLData.groupby('AwayTeam').AS.sum().reset_index()
#print(totalNumberOfShotsPerAwayTeam)

homeConversionRate = (totalNumberOfGoalsPerHomeTeam.FTHG/totalNumberOfShotsPerHomeTeam.HS) * 100
#print(homeConversionRate)
awayConversionRate = (totalNumberOfGoalsPerAwayTeam.FTAG/totalNumberOfShotsPerAwayTeam.AS) * 100
#print(awayConversionRate)


# **Graph that demonstrate home advantage**

# ![image.png](attachment:image.png)
# 
#  
# 

# ![image.png](attachment:image.png)
# 
# from : https://link.springer.com/article/10.1007/s10994-018-5741-1
# 
# 
# The number of matches per number of goals scored by the home (dark grey) and away team (light grey)
# Figure 2 shows the number of matches per number of goals scored by the home (dark grey) and away (light grey) teams. Home teams appear to score more goals than away teams, with home teams having consistently higher frequencies for two or more goals and away teams having higher frequencies for no goal and one goal. Overall, home teams scored 304,918 goals over the whole data set, whereas away teams scored 228,293 goals. In Section 1 of the Supplementary Material, the trend shown in Fig. 2 is also found to be present within each country, pointing towards the existence of a home advantage.
# 
# 
# From this historical statistic, we can also affirm that home teams definitely have an advantage as we can think of the following reasons:
# 
# - Football is a team sport, a cheering crowd helps morale
# - Familarity with pitch and weather conditions helps
# - No need to travel (less fatigue)
# 

# **Data preprocessing**
# 
# This is a supervised learning problem. We have one target variable which is FTR (stands for full-time result). We want to predict FTR given all the features in the dataset. We also want to standarize our dataset and put it on the same scale (mean=0, variance=1) 
# 
# We want to combine or extract certain columns of this dataset to find the final features that we are going to work with.
# 
# features:
# - last 5 games form (in the form of an integer 1.0 representing a Win, 0.5 Draw and 0.0 Loss)
# - Attacking Score (float, based on the number of chances created per game, the number of goals scored and more...)
# - Defending Score (float, based on the number of goals conceded per game, fouls committed, yellow and red cards conceded and more..)
# 
# To do list:
# - find a way to turn free kicks and corner kicks into goal scoring opportunities (how to consider them)?
# - append last 5 games results to model
# - Hyperparameter tuning to increase the performance of our model.

# In[6]:


#Creating new DataFrame that displays characteristics per team with the following features: Home Goals Score, Away Goals Score, Home/Away Attack Strength, Home Goals Conceded, Away Goals Conceded, Home/Away Defensive Strength
columns = ['Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS','HTYCI','ATYCI','HTRCI',
           'ATRCI','HTSOTI','ATSOTI','HTFI','ATFI','HTSI','ATSI','HTCKI','ATCKI','HTCR','ATCR']
season19 = pd.DataFrame(columns=columns)
season19.Team = listOfTeamsInOrder #list of the teams in order 
season19.HGS = total_home_goal_per_team.FTHG #Total home goal scored per team
season19.AGS = total_away_goal_per_team.FTAG #Total Away goal scored per team 
season19.HAS = (season19.HGS/numberOfHomeGamesPerTeam.AwayTeam) / averageHomeTeamGoal #Home team attacking strength
season19.AAS = (season19.AGS/numberOfAwayGamesPerTeam.HomeTeam) / averageAwayTeamGoal #Away team attacking strenth
season19.HGC = total_goal_conceded_per_home_team.FTAG #Total home goal conceded per team 
season19.AGC = total_goal_conceded_per_away_team.FTHG #Total away goal conceded per team 
season19.HDS = (season19.HGC/numberOfHomeGamesPerTeam.AwayTeam) / averageHomeGoalsConceded #Home team defensive strength
season19.ADS = (season19.AGC/numberOfAwayGamesPerTeam.HomeTeam) / averageAwayGoalsConceded #Away team defensive strength 
season19.HTYCI = yellowCardIndexPerHomeTeam
season19.ATYCI = yellowCardIndexPerAwayTeam
season19.HTRCI = redCardIndexPerHomeTeam
season19.ATRCI = redCardIndexPerAwayTeam
season19.HTSOTI = shotOnTargetIndexPerHomeTeam
season19.ATSOTI = shotOnTargetIndexPerAwayTeam
season19.HTFI = foulIndexPerHomeTeam
season19.ATFI = foulIndexPerAwayTeam
season19.HTSI = shotIndexPerHomeTeam
season19.ATSI = shotIndexPerAwayTeam
season19.HTCKI = cornerKicksIndexPerHomeTeam
season19.ATCKI = cornerKicksIndexPerAwayTeam
season19.HTCR = homeConversionRate
season19.ATCR = awayConversionRate
print(season19)

#Important note: The higher a team's HAS/AAS score is, the better they are at scoring goals
#In the other side: The lower a team's HDS/ADS is, the better they are at not conceding goals, which means the stronger they are in defense, the lower their HDS/ADS will be.


# In[7]:


#dataframe with engineered features 

bot = BPLData[['HomeTeam', 'AwayTeam', 'FTR']]

HAS = []
HDS = []
AAS = []
ADS = []
HTYCI = []
ATYCI = []
HTRCI = []
ATRCI = []
HTSOTI = []
ATSOTI = []
HTFI = []
ATFI = []
HTSI = []
ATSI = []
HTCKI = []
ATCKI = []
HTCR = []
ATCR = []


for index, row in bot.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs. 
    HAS.append(season19[season19.Team == row['HomeTeam']]['HAS'].values[0])
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
    
bot['HAS'] =HAS
bot['HDS']=HDS
bot['HTYCI']=HTYCI
bot['HTRCI']=HTRCI
bot['HTSOTI']=HTSOTI
bot['HTFI']=HTFI
bot['HTSI']=HTSI
bot['HTCKI']=HTCKI
bot['HTCR']=HTCR
bot['AAS']=AAS
bot['ADS']=ADS
bot['ATYCI']=ATYCI
bot['ATRCI']=ATRCI
bot['ATSOTI']=ATSOTI
bot['ATFI']=ATFI
bot['ATSI']=ATSI
bot['ATCKI']=ATCKI
bot['ATCR']=ATCR

print(bot)


# In[8]:


#final dataframe 
predictiveBot = BPLData[['HomeTeam','AwayTeam','FTR','HST','AST','HF','AF','HC','AC','HR','AR']]


HAS = []
HDS = []
AAS = []
ADS = []

for index, row in predictiveBot.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs. 
    HAS.append(season19[season19.Team == row['HomeTeam']]['HAS'].values[0])
    HDS.append(season19[season19.Team == row['HomeTeam']]['HDS'].values[0])
    AAS.append(season19[season19.Team == row['AwayTeam']]['AAS'].values[0])
    ADS.append(season19[season19.Team == row['AwayTeam']]['ADS'].values[0])
    
predictiveBot['HAS'] = HAS
predictiveBot['HDS'] = HDS
predictiveBot['AAS'] = AAS
predictiveBot['ADS'] = ADS

print(predictiveBot.head(10))
# FTR : Full Time Result
# HST : Home Shots on Target (goal opportunity)
# AST : Away Shots on Target (goal opportunity)
# HF : Home Faults
# AF : Away Faults
# HC : Home corners (goal opportunity)
# AC : Away corners (goal opportunity)
# HR : Home Red Card (decided to ignore Yellow cards, because it only has an impact on the game if it results to a red card)
# AR : Away Red Card
# HAS : Home Attacking Strength
# HDS : Home Defensive Strength
# AAS : Away Attacking Strength
# ADS : Away Defensive Strength


# In[43]:


#Transforming H,D,A to integers 
def stringToInteger(str):
    if str.FTR == 'H':
        return 1
    elif str.FTR == 'A':
        return -1
    else:
        return 0
   
bot['Result'] = bot.apply(lambda row : stringToInteger(row), axis = 1)
bot


# In[44]:


#Separate into feature set and target variable
#Target is 'Result' (1 == H which means home team wins, 0 == D means Draw,-1 == A means Away team wins)

Y = bot['Result']
x = bot[['HAS', 'HDS', 'HTYCI', 'HTRCI', 'HTSOTI', 'HTFI', 'HTSI',
        'HTCKI', 'HTCR', 'AAS', 'ADS', 'ATYCI', 'ATRCI', 'ATSOTI', 'ATFI', 'ATSI', 'ATCKI', 'ATCR']]

print(x.shape)
print(Y)


# In[45]:


#Standardising the data.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaledData = scaler.fit_transform(x)

print(scaledData)
#print(x)


# In[65]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

#Encoding my target values 
encoder = LabelEncoder()
encoder.fit(bot.FTR)
encoded_Y = encoder.transform(bot.FTR)

y_one_hot_encoded = np_utils.to_categorical(encoded_Y)
print(y_one_hot_encoded)


# In[46]:


#train/test split
X_train, X_test, y_train, y_test = train_test_split(scaledData, Y, test_size=0.20, train_size = 0.80)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


LogisticClassifier = LogisticRegression(solver = 'newton-cg', multi_class= 'multinomial', dual = False, C = 0.75, max_iter = 3) 
y_pred = LogisticClassifier.fit(X_train,y_train).predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[47]:


#Trying out different models

#Trying Logistic Regression 
LogisticClassifier = LogisticRegression(solver = 'newton-cg', multi_class= 'multinomial', dual = False, C = 0.75, max_iter = 3) 
scores = cross_val_score(LogisticClassifier, scaledData, Y, cv=10, scoring = 'accuracy')
print("LogisticRegression accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
y_pred = cross_val_predict(LogisticClassifier, scaledData, Y, cv=10)
conf_mat = confusion_matrix(Y, y_pred)
print(classification_report(Y,y_pred))

clf1 = RandomForestClassifier()
scores = cross_val_score(clf1, scaledData, Y, cv=10, scoring = 'accuracy')
print("Random Forest accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
y_pred1 = cross_val_predict(clf1, scaledData,Y, cv=10)
conf_mat1 = confusion_matrix(Y, y_pred1)

clf3 = LinearSVC()
scores = cross_val_score(clf3, scaledData, Y, cv=10, scoring = 'accuracy')
print("LinearSVC accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
y_pred2 = cross_val_predict(clf3, scaledData, Y, cv=10)
conf_mat2 = confusion_matrix(Y, y_pred2)

#Trying XGBoost
xgBoost = xgb.XGBClassifier()
scores = cross_val_score(xgBoost, scaledData, Y, cv=10, scoring = 'accuracy')
print("XGBoost accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
y_pred3 = cross_val_predict(xgBoost, scaledData, Y, cv=10)
conf_mat3 = confusion_matrix(Y, y_pred3)

#Trying SVM 
svm = SVC(gamma=0.001,C=1, decision_function_shape='ovo')
scores = cross_val_score(svm, scaledData, Y, cv=10)
print("SVC accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
y_pred4 = cross_val_predict(svm, scaledData, Y, cv=10)
conf_mat4 = confusion_matrix(Y, y_pred4)
print(classification_report(Y,y_pred4))

#solver = 'newton-cg', multi_class= 'multinomial'
#gamma='scale', decision_function_shape='ovo'

#Printing confusion matrix:Performance measurement for machine learning classification

print(conf_mat)

print(conf_mat1)

print(conf_mat2)

print(conf_mat3)

print(conf_mat4)




# In[48]:


#Hyperparameter Tuning for logistic regression...
#Using KFold cross validation technique 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=10)
dual = [False]
max_iter = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,145,170,200]
C = [0.0001,0.01,0.1, 0.5, 0.75,1.0,2,3,4,5,6,7,8,9,10,15,20,50,100]
#solver = ['newton-cg','lbfgs','liblinear','sag', 'saga']

param_grid = dict(max_iter = max_iter,C=C,dual = dual)

grid = GridSearchCV(estimator=LogisticClassifier, param_grid=param_grid, cv = 10, n_jobs = -1)

grid_result = grid.fit(scaledData,Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[49]:


Cs = [0.001, 0.01, 0.1, 1, 2,3,5,10,12,15,17,20]
gammas = [0.001, 0.01, 0.1, 1, 'scale']
param_grid = dict(C=Cs,gamma=gammas)
grid_search = GridSearchCV(svm, param_grid, cv=10, n_jobs = -1)
grid_search.fit(scaledData, Y)
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


# In[50]:


def svc_param_selection(X, y, nfolds):
   
    Cs = [0.0001,0.001, 0.01, 0.1, 1, 10]
    gammas = [0.0001,0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

print(svc_param_selection(X_train, y_train, 10))


# In[69]:


seed =7
np.random.seed(seed)


# In[70]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from keras.layers.core import Activation 

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=18, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))  
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience = 10,verbose = 0,mode = 'auto')
    checkpointer = ModelCheckpoint(filepath = "best_weights.hdf5", verbose = 0,save_best_only=True)
                            
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, scaledData, y_one_hot_encoded, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:





# In[ ]:


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=18, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))  
    model.add(Dense(3, activation='softmax'))

                            
    return model
    # Compile model
    
model = baseline_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(scaledData, y_one_hot_encoded, epochs=100, batch_size=10)
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience = 10,verbose = 0,mode = 'auto')
checkpointer = ModelCheckpoint(filepath = "best_weights1.hdf5", verbose = 0,save_best_only=True)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, scaledData, y_one_hot_encoded, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = model.predict(X_test)
print('Your accuracy is:', accuracy_score(y_test, y_pred))


# In[76]:


newDF = pd.DataFrame({'HomeTeam' : 'blablah','AwayTeam':'yolo'}, index=[0])
newDF['newColumn'] = 'blahblhah'
newDF


# In[ ]:




