########################################################################################
###### Este código executa o precesso de seleção e treinamento do algoritmo de ML ######
########################################################################################

###############
# Bibliotecas #
###############

import pandas as pd
import operator
from datetime import date, datetime
from pickle import dump

df = pd.read_csv('..//data/dataset_train/creditcard_ingested.csv')

#########################################
# Preparacao dos dados para treinamento #
#########################################

from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)[['V14','V12','V10']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

###############################################################
# Classificadores que serão utilizados no teste de eficiência #
###############################################################

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

rates = {}

#############################################################################################################################
# Os algoritmos são submetidos a um teste de validação cruzada e é escolhido aquele que teve melhor performance na precicao #
#############################################################################################################################

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    rates[classifier,key] = round(training_score.mean(), 2) * 100


##############################################
# O algoritmo com o melhor resultado é salvo #
##############################################

best_model = max(rates.items(), key=operator.itemgetter(1))[0]

file_name = "{}{}{}_{}".format(date.today().year, date.today().month,date.today().day,best_model[1])

dump(best_model[0], open('..//models/{}.pkl'.format(file_name), 'wb'))

print("\nProcesso de Treinamento Terminado.\n")
