#############################################################################
###### Este código executa o processo de coleta e preparação dos dados ######
#############################################################################

###############
# Bibliotecas #
###############

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, RobustScaler

df = pd.read_csv('..//data/creditcard.csv')

#################################################
# Aplicar scaler nas features 'Amount' e 'Time' #
#################################################

rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)

###########################################################################################################################################################
# Como o dataset não está balanceado, será necessário filtrar os registros da base. Optei por criar uma base onde 50% dos dados são fraudes e 50% não são #
###########################################################################################################################################################

df = df.sample(frac=1)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

#######################################################################################################
#### Tratamento dos dados                                                                             #
## Remoção de outilers das features v14, v12 e v10 que foram selecionadas durante a fase de modelagem #
## A limpeza foi feita por percentil, considerando dados que estão entre 25% e 75%                    #
#######################################################################################################

#v14
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
v14_iqr = q75 - q25
v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

#v12
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25
v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

# v10 
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25
v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

new_df = new_df[['V14','V12','V10','Class']]


new_df.to_csv('..//data/dataset_train/creditcard_ingested.csv',index = False)

print("\nProcesso de Ingest/Data preparation Terminado.\n")
