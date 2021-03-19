######################################################################################################
# Este código funciona como um DAG (directed acyclic graph) para coordenar a execucao de cada código #
######################################################################################################

import os

print("\n ----- Iniciando DAG ---- \n")

os.system("python data_ingest.py")
os.system("python data_train.py")
os.system("python data_predict.py")

print("\n ----- Fim ---- \n")
