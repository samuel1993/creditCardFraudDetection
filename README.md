# creditCardFraudDetection
Operacionalização de um modelo de ML utilizando o dadaset creditCardFraudDetection no Kaggle.
******* O dataset foi reduzido pois o GitHub só aceita arquivos até 100MB.

Para executar o código, basta executar o arquivo /code/dag.py

Arquivos:

### dag.py

Este código funciona como um DAG (directed acyclic graph) para coordenar a execucao de cada código

### data_ingest.py

Este código executa o processo de coleta e preparação dos dados

### data_train.py

Este código executa o precesso de seleção e treinamento do algoritmo de ML

### data_predict.py

Este código executa predição utilizando o ultimo modelo treinado
