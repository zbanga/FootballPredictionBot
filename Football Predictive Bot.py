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

n_features = BPLData.shape[1] #gives the number of features
n_matches = BPLData.shape[0] #gives the number of matches 
#print(str(n_features) + ' features')
#print(str(n_matches) + ' matches ')

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
#print(numberOfAwayGamesPerTeam)


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
plt.show()
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
plt.show()


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

# In[5]:


#Creating new DataFrame that displays characteristics per team with the following features: Home Goals Score, Away Goals Score, Home/Away Attack Strength, Home Goals Conceded, Away Goals Conceded, Home/Away Defensive Strength
season19 = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS'))
season19.Team = listOfTeamsInOrder #list of the teams in order 
season19.HGS = total_home_goal_per_team.FTHG #Total home goal scored per team
season19.AGS = total_away_goal_per_team.FTAG #Total Away goal scored per team 
season19.HAS = (season19.HGS/numberOfHomeGamesPerTeam.AwayTeam) / averageHomeTeamGoal #Home team attacking strength
season19.AAS = (season19.AGS/numberOfAwayGamesPerTeam.HomeTeam) / averageAwayTeamGoal #Away team attacking strenth
season19.HGC = total_goal_conceded_per_home_team.FTAG #Total home goal conceded per team 
season19.AGC = total_goal_conceded_per_away_team.FTHG #Total away goal conceded per team 
season19.HDS = (season19.HGC/numberOfHomeGamesPerTeam.AwayTeam) / averageHomeGoalsConceded #Home team defensive strength
season19.ADS = (season19.AGC/numberOfAwayGamesPerTeam.HomeTeam) / averageAwayGoalsConceded #Away team defensive strength 

print(season19)

#Important note: The higher a team's HAS/AAS score is, the better they are at scoring goals
#In the other side: The lower a team's HDS/ADS is, the better they are at not conceding goals, which means the stronger they are in defense, the lower their HDS/ADS will be.


# In[6]:


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


# In[7]:


#Transforming H,D,A to integers 
def stringToInteger(str):
    if str.FTR == 'H':
        return 1
    elif str.FTR == 'A':
        return -1
    else:
        return 0
   
predictiveBot['Result'] = predictiveBot.apply(lambda row : stringToInteger(row), axis = 1)
#print(predictiveBot)


# In[8]:


#Separate into feature set and target variable
#Target is 'Result' (1 == H which means home team wins, 0 == D means Draw,-1 == A means Away team wins)
Y = predictiveBot['Result']
x = predictiveBot[['HST','AST','HF','AF','HC','AC','HR','AR','HAS','HDS','AAS','ADS']]

#print(x)
#print(Y)


# In[1]:


#Standardising the data.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaledData = scaler.fit_transform(x)

print(scaledData)
#print(x)


# In[19]:


#train/test split
X_train, X_test, y_train, y_test = train_test_split(scaledData, Y, test_size=0.20, train_size = 0.80, random_state = 2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


LogisticClassifier = LogisticRegression(solver = 'newton-cg', multi_class= 'multinomial', dual = False, C = 0.75, max_iter = 3) 
ypred = LogisticClassifier.fit(X_train,y_train).predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[15]:


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
svm = SVC(gamma=0.01,C=10, decision_function_shape='ovo')
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




# In[12]:


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


# In[13]:


# Cs = [0.001, 0.01, 0.1, 1, 2,3,5,10,12,15,17,20]
# gammas = [0.001, 0.01, 0.1, 1, 'scale']
# param_grid = dict(C=Cs,gamma=gammas)
# grid_search = GridSearchCV(svm, param_grid, cv=10, n_jobs = -1)
# grid_search.fit(scaledData, Y)
# print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


# In[ ]:


###try adding data from past seasons

## try adding adding last forms 

#try adding player stats .. ? players strenght ... ? 


# In[ ]:





# In[ ]:




