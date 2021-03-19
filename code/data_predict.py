##############################################################################
###### Este código executa predição utilizando o ultimo modelo treinado ######
##############################################################################

###############
# Bibliotecas #
###############

import os
import pandas as pd
from pickle import load
from datetime import date, datetime

####################################
# Carrega o ultimo modelo treinado #
####################################

model_path = sorted(os.listdir("..//models"))[-1]

model = load(open('..//models/{}'.format(model_path), 'rb'))

####################################################
# Seleciona uma massa de registros para a predicao #
####################################################

df_predic = pd.read_csv('..//data/creditcard.csv')[:500]

X = df_predic[['V14','V12','V10']].values
y = df_predic['Class'].values

######################
# Executa a predicao #
######################

y_predict = model.predict(X)

df_predic['Class'] = y_predict

##############################
# Armazena os dados preditos #
##############################

df_predic.to_csv('../data/predicao/{}{}{}_predicao.csv'.format(date.today().year, date.today().month,date.today().day),index = False)

print("\nA predicao foi feita.\n")
